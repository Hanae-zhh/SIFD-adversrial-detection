
import pickle 
from typing import Sized
import warnings
import os
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import argparse
import numpy as np 
from transformers import BertTokenizer, BertModel
import torch
import nltk
from .attack_instance import read_adv_files
from .refactor import  pre_tokenize

supported_pos_tags = [#nltk词性标注中的一些类别
    #'CC',  # coordinating conjunction, like "and but neither versus whether yet so"  #并列连词
    'JJ',  # Adjective, like "second ill-mannered" #组合式形容词
    'JJR',  # Adjective, comparative, like "colder" #形容词比较级
    'JJS',  # Adjective, superlative, like "cheapest" 形容词最高级
    'NN',  # Noun, singular or mass #名词 单数
    'NNS',  # Noun, plural #名词复数
    'NNP',  # Proper noun, singular #专有名词单数
    'NNPS',  # Proper noun, plural  #专有名词复数
    'RB',  # Adverb#副词
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense 动词过去式
    'VBG',  # Verb, gerund or present participle 动名词/现在分词
    'VBN',  # Verb, past participle #过去分词
    'VBP',  # Verb, non-3rd person singular present #非第三人称单数
    'VBZ',  # Verb, 3rd person singular present #第三人称单数
]

def  get_pos_vocabulary():
    file_name = '/data/zhanghData/AttentionDefense/data/words_pos.dict'
    with open(file_name, 'rb') as f:
            pos_vocabulary = pickle.load(f)
    return pos_vocabulary

def save_pos_vocabulary(vocabulary):
    file_name = '/data/zhanghData/AttentionDefense/data/words_pos.dict'
    with open(file_name,'wb') as f:
        pickle.dump(vocabulary, f)
    print("Successful save pos_vocabulary")


def valid_select(words,sorted_words_idxs,vocabulary,stopwords):
    candidates = [] 
    cand_indices  = []
    # for idx in sorted_words_idxs:
    #     if words[idx] not in vocabulary:
    #         #print(f"{words[idx]} not in vocabulary")
    #         vocabulary[words[idx]] = nltk.pos_tag([words[idx]])[0][1]
    #     if vocabulary[words[idx]] in supported_pos_tags and words[idx] not in stopwords:
    #         candidates.append(words[idx])
    #         cand_indices.append(idx)

    for idx in sorted_words_idxs:
        candidates.append(words[idx])
        cand_indices.append(idx)

    return candidates, cand_indices




def get_importance_words_2(
    model,tokenizer, sentence, pos_vocabulary,stopwords,
    alpha=0.1, max_length=40
    ):
    device = torch.device('cuda')
    model.zero_grad()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    # print("words:{}".format(words))
    # print("keys:{}".format(keys))
    assert len(words) == len(keys)
    input_words = ['[CLS]'] + sub_words + ['[SEP]']
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(input_words)])
    assert len(sub_words) == (len(input_ids[0])-2)
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    output_label = torch.argmax(outputs.logits)
    re_outputs =model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device), labels= output_label)
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
    input_embed = torch.index_select(word_embed,0,input_ids[0].to(device)).to(device)
    input_embed_grad = torch.index_select(word_embed_grad,0,input_ids[0].to(device)).to(device)
    scores_ = torch.mul(input_embed,input_embed_grad).sum(axis =-1)
    scores_ =  torch.nn.functional.softmax(scores_)
    sub_scores= scores_[1:-1].detach().cpu().numpy()
    
    
    scores = []
    for key in keys:
        
        #score = sum(sub_scores[key[0]:key[1]])/(key[1]-key[0])
        score = sum(sub_scores[key[0]:key[1]])
        scores.append(score)
    assert len(words) == len(scores)
    #print('------sorting------')
    # for k,value in zip(words,scores):
    #     print(f"({k},{value})")
    sorted_ = sorted(enumerate(scores),key=lambda x:x[1], reverse=True)
    sorted_words_idxs = [i[0] for i in sorted_]
    candidates, cand_idx = valid_select(words,sorted_words_idxs,pos_vocabulary,stopwords)
    result_num = max(min(len(candidates),int(alpha*len(words))),1)
    return candidates[:result_num], cand_idx[:result_num]
    #        sorted_subs = sorted(zip(all_subs_indices,all_sub_scores),key=lambda x:x[1], reverse=True)

def get_importance_words_v3(
    model,tokenizer, sentence, pos_vocabulary,stopwords,
    alpha=0.1, max_length=40
    ):
    '''

    '''
    device = torch.device('cuda')
    model.zero_grad()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    # print("words:{}".format(words))
    # print("keys:{}".format(keys))
    assert len(words) == len(keys)
    input_words = ['[CLS]'] + sub_words + ['[SEP]']
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(input_words)])
    assert len(sub_words) == (len(input_ids[0])-2)
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    output_label = torch.argmax(outputs.logits)
    
    embeddings_data = getattr(model, 'bert').embeddings.word_embeddings(input_ids.to(device))
    embeddings_data.retain_grad()
    embed_outputs = model(input_ids=None, token_type_ids=None, \
                    attention_mask=(input_ids>0).to(device),inputs_embeds=embeddings_data, labels= output_label)
    #print(f"embed_outputs.keys():{embed_outputs.keys()}") #'loss', 'logits
    embed_outputs.loss.backward()
    #print(f"embedding logit:{embed_outputs.logits}")
    # print(embeddings_data.grad.size())
    scores = torch.mul(embeddings_data,embeddings_data.grad).sum(axis =-1)[0]
    scores = torch.nn.functional.softmax(scores)
    sub_scores  = scores[1:-1].detach().cpu().numpy()
    # sorted_ = sorted(zip(sub_words,scores), key=lambda x:x[1], reverse=True)
    
    # print(sorted_)
    scores = []
    for key in keys:
        
        score = sum(sub_scores[key[0]:key[1]])
        score = sum(sub_scores[key[0]:key[1]])/(key[1]-key[0])
        scores.append(score)
    assert len(words) == len(scores)
    #print('------sorting------')
    # for k,value in zip(words,scores):
    #     print(f"({k},{value})")
    #sorted_words = sorted(zip(words, scores),key=lambda x:x[1], reverse=True)
    #print(f"sorted_words:{sorted_words}")
    sorted_ = sorted(enumerate(scores),key=lambda x:x[1], reverse=True)
    sorted_words_idxs = [i[0] for i in sorted_]
    candidates, cand_idx = valid_select(words,sorted_words_idxs,pos_vocabulary,stopwords)
    #print("After valid:{}".format(candidates))
    result_num = max(min(len(candidates),int(alpha*len(words))),1)
    return candidates[:result_num], cand_idx[:result_num]
    # #        sorted_subs = sorted(zip(all_subs_indices,all_sub_scores),key=lambda x:x[1], reverse=True)
