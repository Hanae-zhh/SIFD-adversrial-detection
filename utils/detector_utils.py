
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
import scipy.stats
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
import time

def get_scores(output,re_outputs):
    scores = []
    label=torch.argmax(output)
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
    return confidence,scores,label,re_labels

def get_word_scores(sub_scores,cand_ids,keys):
    word_scores = []
    for i in range(len(keys)):
        if i in cand_ids:
            if keys[i][0]+1 == keys[i][1]:
                word_scores.append(sub_scores[keys[i][0]])
            else:
                word_scores.append(max(sub_scores[keys[i][0]:keys[i][1]]))
    return word_scores

def sentence_sensitivity_base_information_bert_grad(
    model,tokenizer, sentence,pos_vocabulary,stopwords,
    max_length,batch_size
    ):
    device = torch.device('cuda')
    model.eval()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    #print("-----")
    assert len(words) == len(keys)
    input_words = ['[CLS]'] + sub_words + ['[SEP]']
    #input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(input_words)])
    input_ids = tokenizer.encode(" ".join(words),
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = max_length,           # 设定最大文本长度 200 for IMDb and 40 for sst
                        #pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                    )
    real_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    logit = outputs.logits
    output_label = torch.argmax(logit)
    #print("output label:{}".format(output_label))
    re_outputs =model(input_ids.to(device),token_type_ids=None,attention_mask=(input_ids>0).to(device),\
        labels=output_label)
    re_outputs.loss.backward()
    emb_name = 'word_embeddings'
    for name, param in model.named_parameters():
        #print(name)
        if param.requires_grad and emb_name in name: 
            #print(param.is_leaf) 
            
            word_embed=param.data
            word_embed_grad = param.grad
            #print(param.grad.sum())
            break
    input_embed = torch.index_select(word_embed,0,real_input_ids[0].to(device)).to(device)
    input_embed_grad = torch.index_select(word_embed_grad,0,real_input_ids[0].to(device)).to(device)
    sub_scores = -1* (input_embed_grad).sum(axis =-1)
    sub_scores  = sub_scores.detach().cpu().numpy()    
    #获取有效词，即只测试文本对有效词变化的敏感程度
    candidates, cand_ids = valid_select(words,[i for i in range(len(words))],pos_vocabulary,stopwords)
   
    scores = get_word_scores(sub_scores,cand_ids,keys)

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

    confidences,_,pre_label,re_labels = get_scores(outputs.logits[0],logits)
    #print("pre_label:{}".format(pre_label))
    softmax = torch.nn.functional.softmax(outputs.logits[0])
    orig_softmax =softmax.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy().tolist()
    all_info = []
    for c,s,r in zip(confidences,scores,re_labels):
        all_info.append({'softmax':c,'score':s,'label':r})
    sorted_info= sorted(all_info,key=lambda k:k['score'], reverse=True)
    
    # softmaxs = [info['softmax'] for info in sorted_info]
    # scores = [ info['score'] for info in sorted_info]
    # re_labels = [ info['label'] for info in sorted_info]
    #return orig_softmax,label,softmaxs,scores,re_labels
    return orig_softmax,pre_label,sorted_info

def sentence_sensitivity_base_information_nn_grad(
    model,nn_config, sentence,pos_vocabulary,stopwords):
    device = torch.device('cuda')
    # model.eval()
    input_words = cut_raw(clean_str(sentence, tokenizer=nn_config.spacy_tokenizer),nn_config.max_length)
    input_ = pad(nn_config.max_length, input_words, nn_config.pad_token) 
    input_ids = torch.tensor([prep_seq(input_, nn_config.word_to_idx, nn_config.unk_token)],
                                dtype=torch.int64,).to(device)
    input_embed = model.embedding(input_ids)
    v1 = Variable(input_embed.to(device), requires_grad=True)
    outputs = model(inputs=None,embeddings=v1)
    logit = outputs[0]
    orig_softmax =torch.nn.functional.softmax(logit).detach().cpu().numpy()
    output_label = torch.argmax(logit)
    loss = nn_config.criterion(outputs, output_label.unsqueeze(0))
    loss.backward()
    output_label =  int(output_label)
    input_scores = -1* v1.grad.sum(axis =-1)[0]
    input_scores = input_scores.detach().cpu().numpy()   
   
    #获取有效词，即只测试文本对有效词变化的敏感程度
    candidates, cand_ids = valid_select(input_words,[i for i in range(len(input_words))],pos_vocabulary,stopwords)
    scores = [input_scores[idx]  for idx in range(len(input_scores)) if  idx in cand_ids ]
    
    mask_copies_ids = create_unk(input_ids[0].detach().cpu().numpy(), cand_ids)
    mask_inputs = torch.tensor(mask_copies_ids,dtype=torch.int64,).to(device)
    # print("len of candidates :{}".format(len(cand_ids)))
    # print("len of mask_copies_ids:{}".format(len(mask_copies_ids)))
    with torch.no_grad():
        mask_input_embed = model.embedding(mask_inputs)
        mask_outputs = model(inputs=None,embeddings=mask_input_embed)

    confidences = torch.nn.functional.softmax(mask_outputs,dim=1)
    confidences = confidences.detach().cpu().numpy().tolist()
    re_labels = torch.argmax(mask_outputs,dim=1)
    re_labels = re_labels.detach().cpu().numpy()
    
    all_info = []
    for c,s,r in zip(confidences,scores,re_labels):
        all_info.append({'softmax':c,'score':s,'label':r})
    sorted_info= sorted(all_info,key=lambda k:k['score'], reverse=True)
    
    return orig_softmax,int(output_label),sorted_info  

def create_unk(input_ids, cand_ids):
    mask_inputs_list = []
    for idx in cand_ids:
        inputs = copy.deepcopy(input_ids)
        inputs[idx] = 0
        mask_inputs_list.append(inputs)
    return mask_inputs_list
   
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
    kl_flags 作为特征值
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
    scores_flags-kl_scores
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

