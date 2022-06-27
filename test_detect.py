
from ast import arg
import imp
import pickle
from statistics import mode
import torch 
import random 
import numpy as np 
import time 
import os 
import argparse 
import sys 
import argparse
import scipy.stats
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, AdamW 
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from utils.attack_instance import read_adv_files
from utils.data_utils import pre_tokenize,build_logging
from utils.data_utils import get_pos_vocabulary, get_stopwords
from utils.sensitivity import valid_select, create_mask
from utils.model_utils import load_model,NN_config,load_word_to_id
from utils.data_utils import pad,prep_seq,load_pkl,cut_raw,clean_str
from models.lstm import LSTM
from models.cnn import CNN
from spacy.lang.en import English
from utils.detector_utils import sentence_sensitivity_base_information,sentence_sensitivity_base_information_nn
from utils.detector_utils import sentence_sensitivity_base_information_bert_grad_1,sentence_sensitivity_base_information_nn_grad_1
from utils.detector_utils import sentence_sensitivity_base_information_bert_grad_2,sentence_sensitivity_base_information_nn_grad_2
from utils.detector_utils import feature_extraction_v1,feature_extraction_v2,feature_extraction_v3,feature_extraction_v4

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
    def __init__(self, orig_softmax,pre_label,mask_info):
        self.pre_softmax =  orig_softmax
        self.pre_label = pre_label
        self.mask_softmaxs = [m['softmax'] for m in mask_info]
        self.mask_scores = [m['score'] for m in mask_info]
        self.mask_labels = [m['label'] for m in mask_info]
    def __len__(self):
        return len(self.mask_softmaxs)       

def load_test_set(test_set_path):
    pairs = np.load(test_set_path, allow_pickle=True)
    datas = [i[0] for i in pairs]
    labels = [i[1] for i in pairs]
    return datas,labels

def save_test_set(datas,labels,test_set_path):
    assert len(datas) == len(labels)
    pairs = [(i,j) for i,j in zip(datas,labels)]
    np.save(test_set_path,pairs)

def get_data(atk_instances,mode_,dim,model,tokenizer,nn_config):
    '''
    scores*flag 作为特征值
    '''
    datas = []
    labels = []
    mode_function_map={1:feature_extraction_v1,
                2:feature_extraction_v2,
                3:feature_extraction_v3,
                4:feature_extraction_v4}
    pos_vocabulary = get_pos_vocabulary()
    stopwords = get_stopwords()
    count = 0
    for atk in atk_instances:
        count+=1
        
        if args.score_mode == 1:
            if args.model_type in ['lstm','cnn']:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_nn_grad_1(\
                        model,nn_config, atk.orig_text,pos_vocabulary,stopwords)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_nn_grad_1(\
                        model,nn_config, atk.perd_text,pos_vocabulary,stopwords)
            else:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_bert_grad_1(\
                        model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_bert_grad_1(\
                        model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
        elif args.score_mode == 2:
            '''
            '''
            if args.model_type in ['lstm','cnn']:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_nn_grad_2(\
                        model,nn_config, atk.orig_text,pos_vocabulary,stopwords)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_nn_grad_2(\
                        model,nn_config, atk.perd_text,pos_vocabulary,stopwords)
            else:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_bert_grad_2(\
                        model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_bert_grad_2(\
                        model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
        elif args.score_mode == 0:
            ''''''
            if args.model_type in ['lstm','cnn']:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_nn(\
                        model,nn_config, atk.orig_text,pos_vocabulary,stopwords)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_nn(\
                    model,nn_config, atk.perd_text,pos_vocabulary,stopwords)
            else:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information(\
                    model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information(\
                        model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
        else:
            print("error!")
            break

        normal_info = DetectorInstance(orig_softmax,pre_label,mask_info)
        attack_info = DetectorInstance(perd_orig_softmax,perd_pre_label,perd_mask_info)
        normal_feature = mode_function_map[mode_](normal_info,dim)
        datas.append(normal_feature)
        labels.append(0)
        attack_feature = mode_function_map[mode_](attack_info,dim)
        datas.append(attack_feature)
        labels.append(1)
        print("count:{}/{}".format(count,len(atk_instances)))
    return datas,labels

def test(de_model_path,data,label):
    with open(de_model_path, 'rb') as f:
        model = pickle.load(f)
    print("Sucess load model!")
    predictions = model.predict(data)
    print("----------Test result----------")
    print(classification_report(label, predictions, digits=3))
   
def get_detector_info(model_name):
    infos = model_name.split("/")[-1].split("_")
    print(infos)
    mode_= infos[0][-1]
    dim = infos[2].split(".")[0]
    dim  = dim.split("dim")[1]
    return int(mode_),int(dim)

if __name__ == '__main__':

    ''''''
# CUDA_VISIBLE_DEVICES=1 python3 test_detect.py --score_mode 2 --dataset yelp --model_type roberta --attack deepwordbug
# CUDA_VISIBLE_DEVICES=2 python3 test_detect.py --score_mode 2 --dataset yelp --model_type roberta --attack bae
# CUDA_VISIBLE_DEVICES=0 python3 test_detect.py --score_mode 2 --dataset imdb  --model_type bert --attack textfooler
# CUDA_VISIBLE_DEVICES=1 python3 test_detect.py --score_mode 2 --dataset imdb  --model_type lstm --attack bae
# CUDA_VISIBLE_DEVICES=1 python3 test_detect.py --score_mode 2 --dataset imdb  --model_type lstm --attack deepwordbug

# CUDA_VISIBLE_DEVICES=6 python3 test_detect.py --score_mode 2 --dataset imdb  --model_type lstm --attack textfooler
# CUDA_VISIBLE_DEVICES=5 python3 test_detect.py  --score_mode 2 --dataset imdb  --model_type bert --attack textfooler
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--attack", type=str, default="textfooler")
    argparser.add_argument('--model_type',type=str,default='randomforest')
    argparser.add_argument("--grad",action='store_true', help="Whether use grad when infactoring")
    argparser.add_argument("--score_mode",type=int, default=0)
    args = argparser.parse_args()
    device = torch.device('cuda')


    mode_name_map={1:"scores_flags",
                 2:"scores_kl_flags",
                 3:"kl_flags",
                 4:"scores_flags-kl_scores"}

    
    # if args.grad:
    #    de_model_dir = "/data/zhanghData/AttentionDefense/save_models/detector_grad"
    # else:
    #     de_model_dir = "/data/zhanghData/AttentionDefense/save_models/detector"
    de_model_dir = "/data/zhanghData/AttentionDefense/save_models/detector_scoresmode{}".format(args.score_mode)
    de_model_name = 'imdb_bert/textfooler_0.1/xgboost/mode3_num3000_dim20.pickle'
    de_model_path = os.path.join(de_model_dir,de_model_name)


    mode_,dim = get_detector_info(de_model_name)
    attack_dir = '/data/zhanghData/AttentionDefense/save_results/transfer_attacks'
    if args.dataset in ['imdb','yelp']:
        rm = 0.1
    elif args.dataset == 'agnews':
        rm = 0.2
    attack_path = os.path.join(attack_dir,"{}_{}".format(args.dataset,args.model_type),'base',"{}_{}_500.csv".format(args.attack,rm))
    
    
    test_set_dir =  '/data/zhanghData/AttentionDefense/save_results/test_detector/scoresmode{}'.format(args.score_mode)
    test_set_folder = os.path.join(test_set_dir,"{}_{}".format(args.dataset,args.model_type))
    if not  os.path.exists(test_set_folder):
        os.makedirs(test_set_folder)
    if args.grad:
        test_set_path = os.path.join(test_set_folder,"{}_mode{}_dim{}_grad.npy".format(args.attack,mode_,dim))
    else:
        test_set_path = os.path.join(test_set_folder,"{}_mode{}_dim{}.npy".format(args.attack,mode_,dim))
    
    
    ## get test set 
    if os.path.exists(test_set_path):
        print("load test set")
        datas,labels = load_test_set(test_set_path)
    else:
        # if test set not exist, mask it based on attack file
        
        device = torch.device('cuda')
        max_length = 40 if args.dataset =='sst2' else 200 if args.dataset =='imdb' else 50
        num_labels = 4 if args.dataset == 'agnews' else 2
        model,tokenizer,nn_config=None,None,None
        if args.model_type in ['lstm','cnn']:
            batch_size = 256
        else:
            batch_size = 64 if args.dataset =='sst2' else 16 if args.dataset =='imdb' else 50
        print("load model...")
        load_model_path = '/data/zhanghData/AttentionDefense/save_models/{}_{}/base/best.pt'.format(args.dataset,args.model_type)
        if args.model_type in ['lstm','cnn']:
            word_to_idx, vocab = load_word_to_id(args)
            nn_config = NN_config(len(vocab),num_labels,max_length,word_to_idx)
            if args.model_type == "cnn":
                model = CNN(nn_config)
            else:
                model = LSTM(nn_config)
            model = load_model(load_model_path,model)
            model.cuda()
            
        else:
            if args.model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels, output_attentions=False, output_hidden_states=False)
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
            elif args.model_type == 'roberta':
                model_type = 'roberta-base'
                tokenizer = RobertaTokenizer.from_pretrained(model_type)
                config = RobertaConfig.from_pretrained(model_type, num_labels=num_labels, output_attentions=False, output_hidden_states=False)
                model = RobertaForSequenceClassification.from_pretrained(model_type, config=config) 
            model.cuda()
            model.load_state_dict(torch.load(load_model_path))
            
        atk_instances = read_adv_files(attack_path)
        print("make test set data...")
        datas,labels = get_data(atk_instances,mode_,dim,model,tokenizer,nn_config)
        print("Save test data in :{}".format(test_set_path))
        save_test_set(datas,labels,test_set_path)
    print("de_model_path:{}".format(de_model_path)) 
    test(de_model_path,datas,labels)

