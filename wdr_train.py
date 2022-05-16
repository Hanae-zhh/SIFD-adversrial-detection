
import imp
import logging
import pickle
from pandas import array
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

def load_info(dir_path,dataset,model_type,atk_path):
    normal_file_path = os.path.join(dir_path,"{}_{}".format(dataset,model_type),"normal_{}.npy".format(atk_path))
    attack_file_path = os.path.join(dir_path,"{}_{}".format(dataset,model_type),"attack_{}.npy".format(atk_path))
    normal_info = np.load(normal_file_path, allow_pickle=True)
    attack_info = np.load(attack_file_path, allow_pickle=True)
    return normal_info, attack_info

def get_data(normal_info,attack_info,data_num):
    '''
    scores*flag 作为特征值
    '''

    assert len(normal_info)==len(attack_info)
    assert len(normal_info)>data_num/2
    datas = []
    labels = []
    for num in range(int(data_num/2)):
        
        normal_feature = normal_info[num]
        datas.append(normal_feature)
        labels.append(0)
        attack_feature = attack_info[num]
        datas.append(attack_feature)
        labels.append(1)
    
    # shuffle
    # listpack = list(zip(datas, labels))
    # random.shuffle(listpack)
    # datas[:], labels[:] = zip(*listpack)
    
    return datas,labels

def eval(y,pre):
    logger.info(classification_report(y, pre, digits=3))
    logger.info(confusion_matrix(y, pre))

def xgboost_classifier(x_train,x_test,y_train,y_test,model_path):
    # Create the model using best parameters found
    model = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.34281802,
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

def train( datas,labels,model_path):
    x_train,x_test,y_train,y_test = train_test_split(datas,labels,test_size=0.2, random_state=7)
    xgboost_classifier(x_train,x_test,y_train,y_test,model_path)

if __name__ == '__main__':

    ''''''
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--mode", type=int, default=1)
    argparser.add_argument("--data_num", type=int, default=1000)
    argparser.add_argument("--feat_dim", type=int, default=5)
    argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
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
    
  
    model_dir = "/data/zhanghData/AttentionDefense/save_models/wdr_detector"
    info_dir_path = '/data/zhanghData/AttentionDefense/data/wdr_detector_data'
    atk_names = args.atk_path.split("_")
    print(f"atk_names:{atk_names}")
    model_save_folder = os.path.join(model_dir,"{}_{}".format(args.dataset,args.model_type),"{}_{}".format(atk_names[0],atk_names[1]))
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_path = os.path.join(model_save_folder,"mode{}_num{}_dim{}.pickle".\
                            format(args.mode,args.data_num,args.feat_dim))
    log_path = os.path.join(model_save_folder,"mode{}_num{}_dim{}.log".\
                            format(args.mode,args.data_num,args.feat_dim))

    logger = build_logging(log_path)

    normal_info,attack_info = load_info(info_dir_path,args.dataset,args.model_type,args.atk_path)
    datas,labels = get_data(normal_info,attack_info,args.data_num)
    logging.info("train_data_szie:{}".format(np.array(datas).shape))
    logger.info("Total data number:{}".format(len(datas)))
    train( datas,labels,model_path)



# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000  --atk_path pwws_0.2_2373 
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000  --atk_path textfooler_0.2_1913 
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000  --atk_path deepwordbug_0.2_1938 
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 2000  --atk_path bae_0.2_1154 

# CUDA_VISIBLE_DEVICES=3 python3 wdr_train.py --dataset imdb --data_num 3000 --atk_path pwws_0.1_4157
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb  --data_num 3000  --atk_path textfooler_0.1_3509
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000  --atk_path bae_0.1_1916
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000 --atk_path deepwordbug_0.1_1698

#lstm
# CUDA_VISIBLE_DEVICES=3 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type lstm --atk_path pwws_0.1_3951
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb  --data_num 3000 --model_type lstm --atk_path textfooler_0.1_4059
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type lstm --atk_path bae_0.1_3374
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type lstm --atk_path deepwordbug_0.1_2649

# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type lstm --atk_path pwws_0.2_5026
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type lstm --atk_path textfooler_0.2_5025
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type lstm --atk_path deepwordbug_0.2_4096 
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type lstm --atk_path bae_0.2_3501

# cnn
# CUDA_VISIBLE_DEVICES=3 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type cnn --atk_path pwws_0.1_3884
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb  --data_num 3000 --model_type cnn --atk_path textfooler_0.1_3743
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type cnn --atk_path bae_0.1_3332
# CUDA_VISIBLE_DEVICES=1 python3 wdr_train.py --dataset imdb --data_num 3000 --model_type cnn --atk_path deepwordbug_0.1_3213

# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type cnn --atk_path pwws_0.2_5157
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type cnn --atk_path textfooler_0.2_5103
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type cnn --atk_path deepwordbug_0.2_ 4501
# CUDA_VISIBLE_DEVICES=2 python3 wdr_train.py --dataset agnews --data_num 3000 --model_type cnn --atk_path bae_0.2_3271