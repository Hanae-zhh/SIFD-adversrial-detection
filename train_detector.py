
import imp
import pickle
import torch 
import random 
import numpy as np 
import time 
import os 
import argparse 
import sys 
import argparse
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import AdaBoostClassifier

from utils.data_utils import build_logging
from utils.data_utils import NN_Text
from utils.model_utils import NN_model

class DetectorInstance:
    """
    Aa attacking example for detection.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, all_info_list):
        self.pre_softmax = all_info_list[0]
        self.pre_label = all_info_list[1]
        self.mask_softmaxs = [m['softmax'] for m in all_info_list[2]]
        self.mask_scores = [m['score'] for m in all_info_list[2]]
        self.mask_labels = [m['label'] for m in all_info_list[2]]
    def __len__(self):
        return len(self.mask_softmaxs)       

def load_info(dir_path,dataset,model_type,atk_path):
    normal_file_path = os.path.join(dir_path,"{}_{}".format(dataset,model_type),"normal_{}.npy".format(atk_path))
    attack_file_path = os.path.join(dir_path,"{}_{}".format(dataset,model_type),"attack_{}.npy".format(atk_path))
    normal_info = np.load(normal_file_path, allow_pickle=True)
    attack_info = np.load(attack_file_path, allow_pickle=True)
    nd_instance = []
    ad_instance = []
    for info in normal_info:
        nd_instance.append(DetectorInstance(info))
    for info in attack_info:
        ad_instance.append(DetectorInstance(info))
    return nd_instance,ad_instance

def get_sens_flags(pre_label,re_labels):
    sens_flags =[1 if r!= pre_label else -1 for r in re_labels]
    return sens_flags

def JS_divergence(pre_softmax,mask_softmaxs):
    js_dive = []
    for ms in mask_softmaxs:
        M=(pre_softmax+ms)/2
        js = 0.5*scipy.stats.entropy(pre_softmax,M)+0.5*scipy.stats.entropy(ms, M)
        js_dive.append(js)
    return js_dive

def feature_extraction_v1(info,feat_dim):
    '''
    kl*flag 作为特征值
    '''
    flags = get_sens_flags(info.pre_label,info.mask_labels)
    feature = np.array(flags)*np.array(info.mask_scores)
    
    if feat_dim <= len(feature):
        feature = feature[:feat_dim]
    else:
        feature = np.hstack((feature, np.zeros(feat_dim-len(feature))))
    assert feat_dim == len(feature)
    return feature

def feature_extraction_v2(info,feat_dim):
    '''
    scores*flag 作为特征值
    '''
    flags = get_sens_flags(info.pre_label,info.mask_labels)
    jss = JS_divergence(info.pre_softmax,info.mask_softmaxs)
    feature = np.array(flags)*np.array(info.mask_scores)*np.array(jss)
    
    if feat_dim <= len(feature):
        feature = feature[:feat_dim]
    else:
        feature = np.hstack((feature, np.zeros(feat_dim-len(feature))))
    assert feat_dim == len(feature)
    return feature

def feature_extraction_v3(info,feat_dim):
    '''
    kl*flag 作为特征值
    '''
    flags = get_sens_flags(info.pre_label,info.mask_labels)
    jss = JS_divergence(info.pre_softmax,info.mask_softmaxs)
    feature = np.array(flags)*np.array(jss)
    
    if feat_dim <= len(feature):
        feature = feature[:feat_dim]
    else:
        feature = np.hstack((feature, np.zeros(feat_dim-len(feature))))
    assert feat_dim == len(feature)
    return feature


def feature_extraction_v4(info,feat_dim):
    '''
    kl*flag 作为特征值
    '''
    flags = get_sens_flags(info.pre_label,info.mask_labels)
    jss = JS_divergence(info.pre_softmax,info.mask_softmaxs)
    feature1 = np.array(flags)*np.array(info.mask_scores)
    feature2 = np.array(flags)*np.array(jss)
    assert len(feature1) == len(feature2)
    if int(feat_dim/2) <= len(feature1):
        feature = np.hstack((feature1[:int(feat_dim/2)],feature2[:int(feat_dim/2)]))
    else:
        feature1 = np.hstack((feature1, np.zeros(feat_dim-len(feature1))))
        feature2 = np.hstack((feature2, np.zeros(feat_dim-len(feature2))))
        feature = np.hstack((feature1[:int(feat_dim/2)],feature2[:int(feat_dim/2)]))
    assert feat_dim == len(feature)
    return feature


def get_data(normal_info,attack_info,data_num,feat_dim):
    '''
    scores*flag 作为特征值
    '''
    mode_function_map={1:feature_extraction_v1,
                2:feature_extraction_v2,
                3:feature_extraction_v3,
                4:feature_extraction_v4}

    assert len(normal_info)==len(attack_info)
    assert len(normal_info)>data_num/2
    datas = []
    labels = []
    for num in range(int(data_num/2)):
        
        normal_feature = mode_function_map[args.mode](normal_info[num],feat_dim)
        datas.append(normal_feature)
        labels.append(0)
        attack_feature = mode_function_map[args.mode](attack_info[num],feat_dim)
        datas.append(attack_feature)
        labels.append(1)

    return datas,labels

def eval(y,pre):
    logger.info(classification_report(y, pre, digits=3))
    logger.info(confusion_matrix(y, pre))

def randomforest(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    model = RandomForestClassifier(n_estimators=1600,
                                min_samples_split=10,
                                min_samples_leaf=2,
                                max_features='auto',
                                #max_features='sqrt',
                                max_depth=None, 
                                bootstrap = True)
    model.fit(x_train,y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("sucess saved model!")

    predictions = model.predict(x_train)
    logger.info("----------train eval----------")
    eval(y_train,predictions)
    predictions = model.predict(x_test)
    logger.info("----------train eval----------")
    eval(y_test,predictions)

def xgboost_classifier(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    model = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.1,
                    gamma=0.6770816,
                    min_child_weight=2.5520658,
                    max_delta_step=0.71469694,
                    subsample=0.61460966,
                    colsample_bytree=0.73929816,
                    colsample_bylevel=0.87191725,
                    reg_alpha=0.9064181,
                    reg_lambda=0.5686102,
                    n_estimators=29,
                    silent=0,
                    nthread=4,
                    scale_pos_weight=1.0,
                    base_score=0.5,
                    missing=None,
                  )
    
    model.fit(x_train,y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("sucess saved model!")

    predictions = model.predict(x_train)
    logger.info("----------train eval----------")
    eval(y_train,predictions)
    predictions = model.predict(x_test)
    logger.info("----------test eval----------")
    eval(y_test,predictions)

def lightgbm_classifier(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    parameters = {
    'objective': 'binary',
    'application': 'binary',
    'metric': ['binary_logloss'],
    'num_leaves': 35,
    'learning_rate': 0.05,
    'verbose': 1
    }
    train_data = lgb.Dataset(np.array(x_train), label=np.array(y_train))
    test_data = lgb.Dataset(np.array(x_test), label=np.array(y_test))
    model = lgb.train(parameters,
                        train_data,
                       valid_sets=test_data,
                       num_boost_round=300)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("sucess saved model!")

    predictions = model.predict(x_train)
    logger.info("----------train eval----------")
    eval(y_train,predictions.round())
    predictions = model.predict(x_test)
    logger.info("----------test eval----------")
    eval(y_test,predictions.round())

def svm_classifier(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    model = SVC(C=9.0622635,
          kernel='rbf',
          gamma='scale',
          coef0=0.0,
          tol=0.001,
          probability=True,
          max_iter=-1)
    model.fit(x_train,y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("sucess saved model!")

    predictions = model.predict(x_train)
    logger.info("----------train eval----------")
    eval(y_train,predictions.round())
    predictions = model.predict(x_test)
    logger.info("----------train eval----------")
    eval(y_test,predictions.round())

def perceptron_classifier(x_train,x_test,y_train,y_test,model_path):
    train_ds = NN_Text(np.array(x_train), np.array(y_train))
    train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)

    basic_classifier = NN_model(input_dim=args.feat_dim*1, hidden_dim=50, output_dim=1).to(device)
    c = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(basic_classifier.parameters(), lr=0.001)

    train_loss_history = []
    val_acc_history = []
    iter_per_epoch = len(train_loader)
    num_epochs = 3
    initial_epoch = 1
    log_nth = 2

    for epoch in range(initial_epoch, initial_epoch+num_epochs):
        basic_classifier.train()
        epoch_losses = []
        for i, (data, y_label) in enumerate(train_loader):
            optimizer.zero_grad()
            out = basic_classifier(data.to(device))
            loss = c(out, y_label.to(device))
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (i+1) % log_nth == 0:        
                logger.info ('Epoch [{}/{}], Step [{}/{}], Loss for last {} batches: {:.4f}' 
                        .format(epoch, num_epochs, i+1, iter_per_epoch, log_nth, np.mean(np.array(epoch_losses[-log_nth:]))))
                #print_time()
            
            logger.info ('Epoch [{}/{}] finished with loss = {:.4f}'.format(epoch, num_epochs, np.mean(np.array(epoch_losses))))
        #torch.save(basic_classifier.state_dict(), checkpoints_path+"/final_model_epoch_{}.checkpoint".format(epoch))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    nn_pred = basic_classifier(torch.tensor(x_test.astype('float32')).to(device))
    nn_pred = nn_pred.flatten().detach().cpu().numpy().round()
    print(classification_report(y_test, nn_pred, digits=3))
    print(confusion_matrix(y_test, nn_pred))

def adboost_classifier(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("sucess saved model!")

    predictions = model.predict(x_train)
    logger.info("----------train eval----------")
    eval(y_train,predictions)
    predictions = model.predict(x_test)
    logger.info("----------train eval----------")
    eval(y_test,predictions)



def train( datas,labels,model_path):
    x_train,x_test,y_train,y_test = train_test_split(datas,labels,test_size=0.2, random_state=7)
    if args.detector_type == 'randomforest':
        randomforest(x_train,x_test,y_train,y_test,model_path)
    elif args.detector_type == 'xgboost':
        xgboost_classifier(x_train,x_test,y_train,y_test,model_path)
    elif args.detector_type == 'lightgbm':
        lightgbm_classifier(x_train,x_test,y_train,y_test,model_path)
    elif args.detector_type == 'svm':
        svm_classifier(x_train,x_test,y_train,y_test,model_path)
    elif args.detector_type == 'mnn':
        perceptron_classifier(x_train,x_test,y_train,y_test,model_path)
    elif args.detector_type == 'adboost':
        adboost_classifier(x_train,x_test,y_train,y_test,model_path)


if __name__ == '__main__':

    ''''''
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--mode", type=int, default=1)
    argparser.add_argument("--data_num", type=int, default=1000)
    argparser.add_argument("--feat_dim", type=int, default=5)
    argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
    argparser.add_argument('--detector_type',type=str,default='randomforest')
    argparser.add_argument('--model_type',type=str,default='bert')
    argparser.add_argument("--grad",action='store_true', help='Whether test all examples, if False, test "suss attack" only')
    #argparser.add_argument("--top_k",type=int,default=3)
    #argparser.add_argument("--vote_num",type=int,default=1)
    #argparser.add_argument("--impo_num",type=int,default=5)
    #argparser.add_argument("--alpha",type=float,default=0.2)
    #argparser.add_argument("--use_mask",action='store_true', help="Whether use [MASK] when infactoring")
    
    args = argparser.parse_args()
    device = torch.device('cuda')


    mode_name_map={1:"scores_flags",
                 2:"scores_kl_flags",
                 3:"kl_flags",
                 4:"scores_flags-kl_scores"}
    
    
    if args.grad:
        info_dir_path = '/data/zhanghData/AttentionDefense/data/detector_data_grad'
        model_dir = "/data/zhanghData/AttentionDefense/save_models/detector_grad"
    else:
        model_dir = "/data/zhanghData/AttentionDefense/save_models/detector"
        info_dir_path = '/data/zhanghData/AttentionDefense/data/detector_data'
    atk_names = args.atk_path.split("_")
    print(f"atk_names:{atk_names}")
    model_save_folder = os.path.join(model_dir,"{}_{}".format(args.dataset,args.model_type),"{}_{}".format(atk_names[0],atk_names[1]),args.detector_type)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    # model_path = os.path.join(model_save_folder,"{}_num{}_dim{}.pickle".\
    #                         format(mode_name_map[args.mode],args.data_num,args.feat_dim))
    # log_path = os.path.join(model_save_folder,"{}_num{}_dim{}.log".\
    #                         format(mode_name_map[args.mode],args.data_num,args.feat_dim))

    model_path = os.path.join(model_save_folder,"mode{}_num{}_dim{}.pickle".\
                            format(args.mode,args.data_num,args.feat_dim))
    log_path = os.path.join(model_save_folder,"mode{}_num{}_dim{}.log".\
                            format(args.mode,args.data_num,args.feat_dim))

    logger = build_logging(log_path)

    normal_info,attack_info = load_info(info_dir_path,args.dataset,args.model_type,args.atk_path)
    datas,labels = get_data(normal_info,attack_info,args.data_num,args.feat_dim)
    logger.info("Total data number:{}".format(len(datas)))
    train( datas,labels,model_path)


#randomforest svm xgboost adboost lightgbm

# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 10 --atk_path pwws_0.2_2377 --grad --model_type randomforest
# CUDA_VISIBLE_DEVICES=3 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 10 --atk_path textfooler_0.2_1901 --model_type randomforest
#CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 5 --atk_path deepwordbug_0.2_1922 --model_type xgboost --grad
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset agnews --mode 3 --data_num 2000 --feat_dim 15 --atk_path bae_0.2_1143 --model_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path pwws_0.15_2121 --model_type xgboost --grad

# CUDA_VISIBLE_DEVICES=3 python3 train_detector.py --dataset imdb --mode 1 --data_num 3000 --feat_dim 10 --atk_path pwws_0.1_4124 --model_type randomforest
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path textfooler_0.1_3436 --model_type xgboost --grad 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path bae_0.1_1896 --model_type xgboost --grad
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path deepwordbug_0.1_1661 --model_type xgboost

#-----------------lstm
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path pwws_0.2_4855 --model_type lstm --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path textfooler_0.2_4679 --model_type lstm --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path deepwordbug_0.2_3907 --model_type lstm --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path bae_0.2_3226 --model_type lstm --detector_type xgboost --grad

# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path pwws_0.2_5026 --model_type lstm --detector_type xgboost
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path textfooler_0.2_5025 --model_type lstm --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path deepwordbug_0.2_4096 --model_type lstm --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path bae_0.2_3501 --model_type lstm --detector_type xgboost 

# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path textfooler_0.1_4059 --model_type lstm --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path pwws_0.1_3951 --model_type lstm --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path deepwordbug_0.1_2649 --model_type lstm --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path bae_0.1_3374 --model_type lstm --detector_type xgboost 

#-----------------cnn
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path pwws_0.2_4333 --model_type cnn --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path textfooler_0.2_3972 --model_type cnn --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path deepwordbug_0.2_3709 --model_type cnn --detector_type xgboost --grad
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path bae_0.2_2453 --model_type cnn --detector_type xgboost --grad

# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path pwws_0.2_5157 --model_type cnn --detector_type xgboost
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path textfooler_0.2_5103 --model_type cnn --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path deepwordbug_0.2_4501 --model_type cnn --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=2 python3 train_detector.py --dataset agnews --mode 3 --data_num 3000 --feat_dim 15 --atk_path bae_0.2_3271 --model_type cnn --detector_type xgboost 

# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path textfooler_0.1_3743 --model_type cnn --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path pwws_0.1_3951 --model_type cnn --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path deepwordbug_0.1_3213 --model_type cnn --detector_type xgboost 
# CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --mode 3 --data_num 3000 --feat_dim 5 --atk_path bae_0.1_3332 --model_type cnn --detector_type xgboost 
