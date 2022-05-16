
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
import sys
import copy
import numpy as np 
from transformers import BertTokenizer, BertModel
import torch
import pickle
import nltk
import argparse 
from utils.data_utils import pre_tokenize,get_supported_pos_tags,build_logging

from utils.data_utils import get_pos_vocabulary, get_stopwords,get_supported_pos_tags
from utils.attack_instance import read_adv_files
from utils.model_utils import load_pre_models
supported_pos_tags = get_supported_pos_tags()



def valid_select(words,sorted_words_idxs,vocabulary,stopwords):
    candidates = [] 
    cand_indices  = []
    for idx in sorted_words_idxs:
        if words[idx] not in vocabulary:
            #print(f"{words[idx]} not in vocabulary")
            vocabulary[words[idx]] = nltk.pos_tag([words[idx]])[0][1]
        if vocabulary[words[idx]] in supported_pos_tags and words[idx] not in stopwords:
            candidates.append(words[idx])
            cand_indices.append(idx)
    return candidates, cand_indices

def sensitive_test(words,idx,maxlength,tokenizer,model,device):
    input_ids = tokenizer.encode(
                    " ".join(words[:idx]+'MASK'+words[idx+1:]),
                    truncation=True,                       
                    add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                    max_length = maxlength,           # 设定最大文本长度 200 for IMDb and 40 for sst
                    # pad_to_max_length = True,   # pad到最大的长度  
                    padding = 'max_length',
                    return_tensors = 'pt'       # 返回的类型为pytorch tensor
                )
    output = model(input_ids=input_ids.to(device), token_type_ids=None, \
                    attention_mask=(input_ids>0).to(device))
    label= torch.argmax(output.logits[0])
    return int(label)

def create_mask(words,cand_ids,tokenizer,maxlength):

    all_input_ids  = []
    for i in cand_ids:
        text = copy.deepcopy(words)
        text[i]='[MASK]'
        input_ids = tokenizer.encode(
                        " ".join(text),
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = maxlength,           # 设定最大文本长度 200 for IMDb and 40 for sst
                        # pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                    )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids   


def get_scores(output,re_outputs):
    scores = []
    label=torch.argmax(output,dim=0)
    label=int(label)
    re_labels = torch.argmax(re_outputs,dim=1)
    re_labels = re_labels.detach().cpu().numpy()
    # print("---------In get scores---------")
    # print(f'label:{label}')
    # print(f"re_labels:{re_labels}")
    confidence = torch.nn.functional.softmax(re_outputs,dim=1)
    #print(confidence)
    for prob in confidence:
        #scores.append(float(prob[int(label)]-prob[1-int(label)]))
        scores.append(float(output[label]-prob[label]))
    # print(scores)   
    #print("---------endIn get scores---------")
    return scores,label,re_labels
    #.detach().cpu().numpy()

def words_sensitivity(
    model,tokenizer, sentence,pos_vocabulary,stopwords,
    max_length,batch_size
    ):
    device = torch.device('cuda')
    model.eval()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    #print("words:{}".format(words))
    # print("keys:{}".format(keys))
    assert len(words) == len(keys)
    input_ids = tokenizer.encode(
                        " ".join(words),
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = max_length,           # 设定最大文本长度
                        # pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                   )
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    #获取有效词，即只测试文本对有效词变化的敏感程度
    candidates, cand_ids = valid_select(words,[i for i in range(len(words))],pos_vocabulary,stopwords)
    if len(candidates) ==0:
        candidates = words
        cand_ids = [i for i in range(len(words))]
    mask_copies_ids = create_mask(words, cand_ids, tokenizer,max_length)
    #print(f"mask_copies_ids.szie:{mask_copies_ids.size()}")
    mask_dataloader = DataLoader(mask_copies_ids, batch_size = batch_size, shuffle = False)
    logits = []
    for i, batch in enumerate(mask_dataloader):
            with torch.no_grad():
                re_outputs = model(input_ids=batch.to(device), token_type_ids=None, \
                    attention_mask=(batch>0).to(device))
                logits.append(re_outputs.logits)
    logits = torch.cat(logits, dim=0)
    # print("logits.szie:{}".format(logits.size()))
    # print("len of mask_copies_ids:{}".format(len(mask_copies_ids)))
    scores,pre_label,re_labels = get_scores(outputs.logits[0],logits)
    return  words,candidates,cand_ids,scores,pre_label,re_labels


# def sentence_sensitivity(model,tokenizer, sentence, label, pos_vocabulary,stopwords,
#     vote_num,alpha, max_length,batch_size
#     ):
#     #print("-------------In sentence sensitivity-------------")
#     words,candidates,cand_ids,scores,pre_label,re_labels = words_sensitivity(
#                 model,tokenizer, sentence, pos_vocabulary,stopwords,
#                 max_length,batch_size)
#     #print( "label:{},pre_label:{}".format(label,pre_label))
#     pre_acc_flag = 1 
#     sens_flag = 0
#     if label != pre_label:
#         #print("Error for predict!")
#         pre_acc_flag = 0
#         return 0,0
        
#     scores_labels_sorted_ = sorted(zip(scores,re_labels),key=lambda k:k[0], reverse=True)
#     #print(scores_labels_sorted_)
#     #vote_num = min(max(int(alpha*len(sentence.split(" "))),1),vote_num)
#     vote_num = min(len(scores_labels_sorted_),vote_num)
#     vote_labels = [k[1] for k in scores_labels_sorted_][:vote_num]
#     #print(f"vote_labels:{vote_labels}")
#     # print("int(label):{},int(sum(re_labels)/len(re_labels)>=0.5):{}"\
#     #     .format(int(label),int(sum(re_labels)/len(re_labels)>=0.5)))
#     if len(vote_labels) == 0:
#          print(sentence)
#          print(vote_num)
#          print(scores_labels_sorted_)
#     else:
#         if int(label) != int(sum(vote_labels)/len(vote_labels)>=0.5):
#             sens_flag = 1
#     #print("sensitivity flag:{}".format(sens_flag)) 
#     return pre_acc_flag,sens_flag


def sensitivity_and_importance(model,tokenizer, sentence, pos_vocabulary,stopwords,
    vote_num,alpha, max_length,batch_size
    ):
    #print("-------------In sentence sensitivity-------------")
    words,candidates,cand_ids,scores,pre_label,re_labels = words_sensitivity(
                model,tokenizer, sentence, pos_vocabulary,stopwords,
                max_length,batch_size)
    #print( "label:{},pre_label:{}".format(label,pre_label))
    '''------for sensitivity------'''
    
    sens_flag = 0
    scores_labels_sorted_ = sorted(zip(scores,re_labels),key=lambda k:k[0], reverse=True)
    #print("scores_labels_sorted_")
    #vote_num = min(max(int(alpha*len(candidates)),1),vote_num)
    vote_num = min(len(scores_labels_sorted_),vote_num)
    vote_labels = [k[1] for k in scores_labels_sorted_][:vote_num]
    actual_num = len(vote_labels) 
    #print(f"vote_labels:{vote_labels}")
    # print("int(label):{},int(sum(re_labels)/len(re_labels)>=0.5):{}"\
    #     .format(int(label),int(sum(re_labels)/len(re_labels)>=0.5)))
    if int(pre_label) != int(sum(vote_labels)/len(vote_labels)>=0.5):
            sens_flag = 1


    '''------for importance words------'''
    scores_candids_sorted_ = sorted(zip(scores,cand_ids),key=lambda k:k[0], reverse=True)
    #importance_num = max(min(len(candidates),int(alpha*len(words))),1)
    importance_num = min(min(len(candidates),int(alpha*len(words))),5)
    importance_ids = [k[1] for k in scores_candids_sorted_[:importance_num]]
    importance_words  = [words[i] for i in importance_ids]
    actual_im_num = len(importance_words)

    return pre_label,sens_flag,importance_ids,importance_words, actual_num, actual_im_num