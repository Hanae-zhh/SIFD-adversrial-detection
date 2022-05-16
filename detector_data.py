
from cProfile import label
from typing import Sized
import warnings
import os
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import  BertForMaskedLM
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader, random_split 
from transformers import BertForSequenceClassification, AdamW 
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup 
import argparse
import copy
import numpy as np 
from transformers import BertTokenizer, BertModel
import torch
import pickle
import nltk
import sys
from torch.autograd import Variable
from utils.attack_instance import read_adv_files
from utils.data_utils import pre_tokenize
from utils.data_utils import get_pos_vocabulary, get_stopwords
from utils.attack_instance import read_adv_files
from utils.sensitivity import valid_select, create_mask
from utils.model_utils import load_model,NN_config,load_word_to_id
from utils.data_utils import pad,prep_seq,load_pkl,cut_raw,clean_str
from models.lstm import LSTM
from models.cnn import CNN
from spacy.lang.en import English
from utils.detector_utils import sentence_sensitivity_base_information,sentence_sensitivity_base_information_nn
from utils.detector_utils import sentence_sensitivity_base_information_bert_grad,sentence_sensitivity_base_information_nn_grad
import time

if __name__ == '__main__':

    ''''''
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--model_type", type=str, default='bert')
    #argparser.add_argument("--attack_model", type=str, default='bert')
    #argparser.add_argument("--victim_model", type=str, default='bert')
    argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
    argparser.add_argument("--grad",action='store_true', help="Whether use [MASK] when infactoring")
    #argparser.add_argument("--top_k",type=int,default=3)
    #argparser.add_argument("--vote_num",type=int,default=1)
    #argparser.add_argument("--impo_num",type=int,default=5)
    #argparser.add_argument("--alpha",type=float,default=0.2)

    #argparser.add_argument("--test_all",action='store_true', help='Whether test all examples, if False, test "suss attack" only')
    args = argparser.parse_args()
    device = torch.device('cuda')
   
    max_length = 40 if args.dataset =='sst2' else 200 if args.dataset =='imdb' else 50
    num_labels = 4 if args.dataset == 'agnews' else 2
    if args.model_type in ['lstm','cnn']:
        batch_size = 256
    else:
        batch_size = 64 if args.dataset =='sst2' else 16 if args.dataset =='imdb' else 50
    
    load_model_path = '/data/zhanghData/AttentionDefense/save_models/{}_{}/base/best.pt'.format(args.dataset,args.model_type)
    if 'bert' in args.model_type :
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels, output_attentions=False, output_hidden_states=False)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        model.cuda()
        model.load_state_dict(torch.load(load_model_path))
    else:
        word_to_idx, vocab = load_word_to_id(args)
        nn_config = NN_config(len(vocab),num_labels,max_length,word_to_idx)
        if args.model_type == "cnn":
            model = CNN(nn_config)
        else:
            model = LSTM(nn_config)
        model = load_model(load_model_path,model)
        model.cuda()
        
    pos_vocabulary = get_pos_vocabulary()
    stopwords = get_stopwords()
    ##攻击文件导入
    atk_path = '/data/zhanghData/AttentionDefense/save_results/transfer_attacks/{}_{}/base/{}.csv'.\
        format(args.dataset,args.model_type,args.atk_path)
    atk_instances = read_adv_files(atk_path)


    x = []
    y = []
    normal_all_info = []
    attack_all_info = []
    start_time = time.time()
    count = 0 
    attack_names = args.atk_path.split("_")
    for atk in atk_instances:
        count += 1
        print("process information----count{}/{}".format(count,attack_names[2]))
        # orig_softmax,pre_label,softmaxs,scores,re_labels=sentence_sensitivity_base_information(\
        #         model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
        if args.grad:
            if args.model_type in ['lstm','cnn']:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_nn_grad(\
                        model,nn_config, atk.orig_text,pos_vocabulary,stopwords)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_nn_grad(\
                        model,nn_config, atk.perd_text,pos_vocabulary,stopwords)
            else:
                orig_softmax,pre_label,mask_info=sentence_sensitivity_base_information_bert_grad(\
                        model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
                #print(scores,pre_label,re_labels)
                perd_orig_softmax,perd_pre_label,perd_mask_info=sentence_sensitivity_base_information_bert_grad(\
                        model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
        else:
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
        #print("atk.ground:{}".format(atk.ground))
        if pre_label != atk.ground or perd_pre_label == atk.ground:
                continue
        normal_all_info.append((orig_softmax,pre_label,mask_info))
        attack_all_info.append((perd_orig_softmax,perd_pre_label,perd_mask_info))
    end_time = time.time()
    
    if args.grad:
        dir_path = '/data/zhanghData/AttentionDefense/data/detector_data_grad'
    else:
        dir_path = '/data/zhanghData/AttentionDefense/data/detector_data'
    info_length = len(normal_all_info)
    save_folder = os.path.join(dir_path,"{}_{}".format(args.dataset,args.model_type))
    if not  os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    normal_file_path = os.path.join(save_folder,"normal_{}_{}_{}.npy".format(attack_names[0],attack_names[1],info_length))
    attack_file_path = os.path.join(save_folder,"attack_{}_{}_{}.npy".format(attack_names[0],attack_names[1],info_length))
    np.save(normal_file_path,normal_all_info)
    np.save(attack_file_path,attack_all_info)
    print("save total info:{} !".format(info_length))
    print("Time Consumption:{:.2f}min".format((end_time-start_time)/60))

    # CUDA_VISIBLE_DEVICES=0 python3 detector_data.py --dataset agnews --model_type cnn --atk_path textfooler_0.2_7600 --grad
    # CUDA_VISIBLE_DEVICES=2 python3 detector_data.py --dataset agnews --atk_path deepwordbug_0.2_7600 --grad
    #
    # CUDA_VISIBLE_DEVICES=1 python3 detector_data.py --dataset imdb --model_type cnn --atk_path bae_0.1_5000 --grad
    # CUDA_VISIBLE_DEVICES=0 python3 detector_data.py --dataset imdb --atk_path textfooler_0.1_5000 --grad
    # CUDA_VISIBLE_DEVICES=1 python3 detector_data.py --dataset imdb --atk_path bae_0.1_3000 --grad


    # CUDA_VISIBLE_DEVICES=2 python3 detector_data.py --dataset imdb --model_type bert --atk_path bae_0.1_3000 
    # CUDA_VISIBLE_DEVICES=2 python3 detector_data.py --dataset imdb --model_type bert --atk_path deepwordbug_0.1_3000 
