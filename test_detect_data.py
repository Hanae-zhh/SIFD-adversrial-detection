
from typing import Sized
import warnings
import os
import torch
import torch.nn as nn
import json
import sys 
import argparse
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import  BertForMaskedLM
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader, random_split 
from transformers import BertForSequenceClassification, AdamW 
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup 

import copy
import numpy as np 
from transformers import BertTokenizer, BertModel
import torch
import pickle
import nltk
from models.lstm import LSTM
from models.cnn import CNN
from utils.model_utils import load_model,NN_config,load_word_to_id

from utils.attack_instance import AttackInstance
from utils.data_utils import pad,prep_seq,load_pkl,cut_raw,clean_str
import csv


def eval_single(sentence):
    model.eval()
    if args.model_type in ['lstm','cnn']:
        input_words = cut_raw(clean_str(sentence, tokenizer=nn_config.spacy_tokenizer),nn_config.max_length)
        input_ = pad(nn_config.max_length, input_words, nn_config.pad_token) 
        input_ids = torch.tensor([prep_seq(input_, nn_config.word_to_idx, nn_config.unk_token)],
                                    dtype=torch.int64,).to(device)
        outputs = model(inputs=input_ids,embeddings=None)
        output_label = torch.argmax(outputs[0]) 
    else:
        input_ids = tokenizer.encode(sentence,
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = max_length,           # 设定最大文本长度 200 for IMDb and 40 for sst
                        #pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                    )
        outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
        output_label = torch.argmax(outputs.logits[0])
    return int(output_label)

def test_instance(instance):
    model.eval()
    pre_orig = eval_single(instance.orig_text)
    pre_perd = eval_single(instance.perd_text)
    
    if pre_orig == instance.ground and pre_perd!= instance.ground:
        return True
    else:
        return False

def filter_adv_files(file_path, b_flag = False):
    b_flag = False
    filtered_lines = []
    print("filtering...")
    with open(file_path, mode='r') as csvf:
        csv_reader = csv.DictReader(csvf)
        line_count  = 0
        filtered_count = 0
        for line in csv_reader:
            line_count += 1            
            instance=AttackInstance(ground_truth=line['ground_truth_output'],\
                            orig_text=line['original_text'],orig_label=line['original_output'],\
                            perd_text=line['perturbed_text'], perd_label=line['perturbed_output'],\
                            orig_score=line['original_score'], perd_score=line['perturbed_score'], \
                            result_type=line['result_type'], num_queries=line['num_queries']
                            ,b_flag=b_flag)
            flag = test_instance(instance)
            if flag:
                filtered_count+=1
                filtered_lines.append(line)
            if filtered_count >= 500:
                break
    print("Read lines:{},flited_lines:{}, Save lines:{}".format(line_count,len(filtered_lines),len(filtered_lines[:500])))
    return filtered_lines[:500]

def save_csv(file_path,lines):
    fieldnames = list(lines[1].keys())
    with open(file_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)
    print("sucessful saved filtered attack files!")



if  __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--atk_path", type=str, default="")
    argparser.add_argument('--model_type',type=str,default='randomforest')
    args = argparser.parse_args()
    device = torch.device('cuda')

    if args.dataset =='sst2':
        max_length = 40 
    elif args.dataset == 'imdb':
        max_length = 200
    elif args.dataset == 'agnews':
        max_length = 50
    elif args.dataset == 'yelp':
        max_length = 200
    num_labels = 4 if args.dataset == 'agnews' else 2
    model,tokenizer,word_to_idx=None,None,None
    if args.model_type in ['lstm','cnn']:
        batch_size = 256
    else:
        batch_size = 64 if args.dataset =='sst2' else 32
    print("batch_size:{},max_length:{},number_class:{}".format(batch_size,max_length,num_labels))
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
    ##攻击文件导入
    attack_dir = '/data/zhanghData/AttentionDefense/save_results/transfer_attacks'
    attack_file_name = args.atk_path
    names = attack_file_name.split("_")
    attack_path = os.path.join(attack_dir,"{}_{}".format(args.dataset,args.model_type),'base',"{}.csv".format(attack_file_name))
    print("Read attack file from :{}".format(attack_path))
    
    filter_lines = filter_adv_files(attack_path, b_flag = False)
    save_file_name = "{}_{}_{}".format(names[0],names[1],len(filter_lines))
    filtered_save_path = os.path.join(attack_dir,"{}_{}".format(args.dataset,args.model_type),'base',"{}.csv".format(save_file_name))
    print("Save flitered attack file in :{}".format(filtered_save_path))
    save_csv(filtered_save_path,filter_lines)

# CUDA_VISIBLE_DEVICES=2 python3 test_detect_data.py --dataset imdb --model_type cnn --atk_path deepwordbug_0.1_5000 
# CUDA_VISIBLE_DEVICES=2 python3 test_detect_data.py --dataset imdb --atk_path deepwordbug_0.1_3000 --model_type bert
# CUDA_VISIBLE_DEVICES=1 python3 test_detect_data.py --dataset yelp --model_type lstm --atk_path depwordbug_0.1_3000 
# CUDA_VISIBLE_DEVICES=3 python3 test_detect_data.py --dataset yelp --model_type roberta --atk_path pwws_0.1_5000 