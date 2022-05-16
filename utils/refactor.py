from asyncio.log import logger
from multiprocessing import reduction
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
import spacy
from data_utils import cos_simlarity,get_stopwords


def get_vocab():

    nlp = spacy.load("en_core_web_sm")
    vocabulary = list(nlp.vocab.strings)
    return vocabulary

def pre_tokenize(seq, tokenizer, use_MASK=False, mask_indices=[], max_length=0):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')
    
    if use_MASK:
        for idx in mask_indices:
            words[idx] = '[MASK]'

    
    sub_words =  []
    keys = []
    index = 0
    if max_length == 0:
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
    else:
        new_words =  []
        for word in words:
            sub = tokenizer.tokenize(word)
            if len(sub_words)+len(sub) > max_length-2:
                break
            new_words.append(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        words = new_words

    if words[-1] =='':
        words = words[:-1]
        keys = keys[:-1]

    return words, sub_words, keys

def get_replace_word(orig_word,substitutes,sub_scores,mlm_model,tokenizer,vocabulary, stop_words,use_bpe=True,glove_dict={}):
    sub_len,top_k = substitutes.size()
    if sub_len == 0:
        #print("error for len=0 of sub-words")
        return " ", []
        
    elif sub_len == 1:
        for idx in substitutes[0]:
            word = tokenizer._convert_id_to_token(int(idx))
            if word in vocabulary and word not in stop_words and "##" not in word and len(word)>1 :
                if cos_simlarity(glove_dict,orig_word,word)>0.5:
                    return word,[idx]
        #print("error for len=1, all words are stop-words")
        return " ", []

    else:
        all_subs_indices = [[int(i)] for i in substitutes[0]]
        all_sub_scores = [i for i in sub_scores[0]]
        k=1
        for subs in substitutes[1:]:
            temp_subs = []
            temp_scores = []
            for  i in range(len(all_subs_indices)):
                for  j in range(len(subs)):
                    temp_subs.append(all_subs_indices[i] + [int(subs[j])])
                    temp_scores.append(all_sub_scores[i]+sub_scores[k][j])
            all_subs_indices = temp_subs
            all_sub_scores = temp_scores
            k += 1
        sorted_subs = sorted(zip(all_subs_indices,all_sub_scores),key=lambda x:x[1], reverse=True)
        all_subs_indices = [i[0] for i in sorted_subs]
        for sub_indices in all_subs_indices:
            tokens = [tokenizer._convert_id_to_token(i) for i in sub_indices]
            word = tokenizer.convert_tokens_to_string(tokens)
            if word in vocabulary and word not in stop_words and "##" not in word and len(word)>1 :
                if cos_simlarity(glove_dict,orig_word,word)>0.5:
                    return word, sub_indices
        #print("error for len>1, all words are not fitting the conditions")
        return " ", []

def ppl_score(input_ids,c_loss,mlm_model):
    inputs_tensor = torch.tensor(input_ids)
    sbu_predictions = mlm_model(inputs_tensor.to('cuda'))[0]
    ppl = c_loss(sbu_predictions, inputs_tensor) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl)) # N
    return ppl

def perplexity(mlm_model, sub_input_ids, replace_indices, idx, keys):
    #print(f"keys[idx]:{keys[idx]}")
    assert len(replace_indices) == (keys[idx][1]-keys[idx][0])
    new_sub_input_ids = copy.deepcopy(sub_input_ids)
    count = 0
    for i in range(keys[idx][0],keys[idx][1]):
        new_sub_input_ids[0][i]= int(replace_indices[count].cpu())
        count += 1
    #print(f"new_sub_input_ids:{new_sub_input_ids}")
    c_loss = nn.CrossEntropyLoss(reduction='none')
    score = ppl_score(sub_input_ids,c_loss,mlm_model)
    new_score = ppl_score(new_sub_input_ids,c_loss,mlm_model)
    if score >= new_score:
        return False
    else:
        return True 

def refactor_words(mlm_model,tokenizer,\
    sentence, words_indices, max_length=512, top_k=10,\
    vocabulary=[], stop_words=[],usemask=False,glove_model={}
    ):
    seq = sentence.replace('\n', '').lower()
    orig_words = seq.split(' ')
    #words, sub_words, keys = pre_tokenize(sentence, tokenizer)
    words, sub_words, keys = pre_tokenize(sentence, tokenizer, usemask, words_indices)
    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    sub_input_ids = [tokenizer.convert_tokens_to_ids(sub_words)]
    #print("sub_input_ids:{}".format(sub_input_ids))
    words_predicted = mlm_model(torch.tensor(sub_input_ids).to('cuda'))[0].squeeze()  # sub_length, vocab
    #print(f"the size of word_predictions: {words_predicted.shape}")#11,30522
    word_pred_scores, words_pred = torch.topk(words_predicted, top_k, -1)  # ub_length, k
    # print(f"the size of word_pred_scores_all: {word_pred_scores.shape}")#
    # print(f"the size of word_predictions: {words_pred.shape}")#
    word_pred_scores  = word_pred_scores[1:len(sub_words) + 1, :]
    words_pred = words_pred[1:len(sub_words) + 1, :]
    new_sent = copy.deepcopy(sentence)
    new_sent = new_sent.split(" ")
    changes = []
    count = 0
    for idx in words_indices:
        substitutes =  words_pred[keys[idx][0]:keys[idx][1],:]
        sub_scores = word_pred_scores[keys[idx][0]:keys[idx][1],:]
        replace_word,replace_indices = get_replace_word(orig_words[idx],substitutes,sub_scores,mlm_model,tokenizer,\
            vocabulary,stop_words,use_bpe=True,glove_dict=glove_model)
        if replace_word == " " :
            replace_word=new_sent[idx]
            count += 1
        # print("replace_word:{}".format(replace_word))
        # print("replace_indices:{}".format(replace_indices))
        # print(f"words[idx]:{words[idx]}")
        # if replace_word != words[idx]:
        #     print("Not ==")
        #     replace_flag = perplexity(mlm_model, sub_input_ids, replace_indices,idx,keys )
        #     if not replace_flag:
        #         replace_word = words[idx]
        # print("replace_word after ppl:{}".format(replace_word))

        #print("replace_word [{}] for {}".format(replace_word,idx))
        changes.append((new_sent[idx],replace_word))
        new_sent[idx]=replace_word
    return " ".join(new_sent), changes, count

# if __name__ == '__main__':
#     sentence = 'I am happy , and I like this movie .'
#     words_indices = [2,8]

#     stop_words = get_stopwords()
#     vocabulary = get_vocab()

#     config_atk = BertConfig.from_pretrained("bert-base-uncased")
#     mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config_atk)
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
#     mlm_model.to('cuda')

#     new_sentence,changes,change_flag = refactor_words(mlm_model,tokenizer,
#         sentence,words_indices,max_length=20,top_k=3,\
#         vocabulary=vocabulary,stop_words=stop_words,usemask=False)
#     print("new sentence:{}\nchanges:{}".format(new_sentence, changes))

    # CUDA_VISIBLE_DEVICES=1 python3 infactor.py 

