
import pickle
from tokenizers import Tokenizer
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

from utils.attack_instance import read_adv_files
from utils.data_utils import pre_tokenize,build_logging
from utils.data_utils import get_pos_vocabulary, get_stopwords
from utils.sensitivity import valid_select, create_mask
from utils.detect import load_info,get_data
from utils.model_utils import load_pre_models
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':

    ''''''
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--mode", type=int, default=1)
    argparser.add_argument("--data_num", type=int, default=1000)
    argparser.add_argument("--feat_dim", type=int, default=5)
    argparser.add_argument('--train_file',type=str,default="")
    argparser.add_argument('--model_type',type=str,default='randomforest')
    argparser.add_argument('--target_model',type=str,default='bert')
    argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
    #argparser.add_argument("--top_k",type=int,default=3)
    #argparser.add_argument("--vote_num",type=int,default=1)
    #argparser.add_argument("--impo_num",type=int,default=5)
    #argparser.add_argument("--alpha",type=float,default=0.2)
    #argparser.add_argument("--use_mask",action='store_true', help="Whether use [MASK] when infactoring")
    #argparser.add_argument("--test_all",action='store_true', help='Whether test all examples, if False, test "suss attack" only')
    args = argparser.parse_args()
    device = torch.device('cuda')

    num_labels = 4 if args.dataset == 'agnews' else 2

    mode_name_map={1:"scores_flags",
                 2:"scores_kl_flags",
                 3:"kl_flags",
                 4:"scores_flags-kl_scores"}

    target_model_dir = '/data/zhanghData/AttentionDefense/save_models'
    model_path = os.path.join(target_model_dir,"{}_{}".format(args.dataset,args.target_model),"base","best.pt")
    if args.target_model in ['bert', 'roberta']:
        target_model,tokenizer = load_pre_models(args.target_model,num_labels,dropout=0.1)


    info_dir_path = '/data/zhanghData/AttentionDefense/data/detector_data'
    normal_info,attack_info = load_info(info_dir_path,args.dataset,args.atk_path)
    test_datas,test_labels = get_data(normal_info,attack_info,args.data_num,args.feat_dim,args.mode)
    
    atk_names = args.atk_path.split("_")
    model_dir = "/data/zhanghData/AttentionDefense/save_models/detector"
    model_save_folder = os.path.join(model_dir,args.dataset,args.train_file,args.model_type)
    model_path = os.path.join(model_save_folder,"{}_num{}_dim{}.pickle".\
                        format(mode_name_map[args.mode],args.data_num,args.feat_dim))

    with open (model_path,'rb') as f:
        detector = pickle.load(f)
    print("sucess laod detector!")
    predictions = detector.predict(test_datas)
    
    print(classification_report(test_labels, predictions, digits=3))
    print(confusion_matrix(test_labels, predictions))
    
# CUDA_VISIBLE_DEVICES=3 python3 detector_test.py --dataset agnews --train_file pwws_0.2 --mode 1 --data_num 3000 --feat_dim 10  --model_type randomforest --atk_path textfooler_0.2_1901

#CUDA_VISIBLE_DEVICES=3 python3 detector_test.py --dataset imdb --train_file pwws_0.1 --mode 1 --data_num 3000 --feat_dim 10  --model_type randomforest --atk_path textfooler_0.1_3436