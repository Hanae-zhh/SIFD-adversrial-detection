
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
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
from transformers import BertTokenizer, BertModel
import torch
import pickle
import nltk
import sys
import scipy

from utils.data_utils import pre_tokenize
from utils.sensitivity import valid_select, create_mask


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

def sentence_sensitivity_base_information(
    model,tokenizer, sentence,pos_vocabulary,stopwords,
    max_length,batch_size
    ):
    device = torch.device('cuda')
    model.eval()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    #print("words:{}".format(words))
    # print("keys:{}".format(keys))
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
    # assert len(sub_words) == (len(input_ids[0])-2)
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    
    #获取有效词，即只测试文本对有效词变化的敏感程度
    candidates, cand_ids = valid_select(words,[i for i in range(len(words))],pos_vocabulary,stopwords)
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

    confidences,scores,pre_label,re_labels = get_scores(outputs.logits[0],logits)
    
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

def load_info(dir_path,dataset,atk_path):
    normal_file_path = os.path.join(dir_path,dataset,"normal_{}.npy".format(atk_path))
    attack_file_path = os.path.join(dir_path,dataset,"attack_{}.npy".format(atk_path))
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


def get_data(normal_info,attack_info,data_num,feat_dim,mode):
    '''
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
        
        normal_feature = mode_function_map[mode](normal_info[num],feat_dim)
        datas.append(normal_feature)
        labels.append(0)
        attack_feature = mode_function_map[mode](attack_info[num],feat_dim)
        datas.append(attack_feature)
        labels.append(1)

    return datas,labels

