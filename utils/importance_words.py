
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

    
def encode_fn(tokenizer, text_list):
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        #max_length = max_length,           # 设定最大文本长度 200 for IMDb and 40 for sst
                        # pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                    )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids
    
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

def get_importance_words_v1(
    model,tokenizer, sentence, pos_vocabulary,stopwords,
    alpha=0.1, attention_layer = 11, max_length=40
    ):
    '''
    attention指导重要单词的获取
    '''
    device = torch.device('cuda')
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    
    assert len(words) == len(keys)
    input_words = ['[CLS]'] + sub_words + ['[SEP]']
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(input_words)])
    # input_ids = tokenizer.encode(
    #                     " ".join(words),
    #                     truncation=True,                       
    #                     add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
    #                     return_tensors = 'pt'       # 返回的类型为pytorch tensor
    
    #                 )
    #print(f"len(sub_words):{len(sub_words)}")
    #print(f"len(input_ids[0]):{len(input_ids[0])}")


    assert len(sub_words) == (len(input_ids[0])-2)
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device),output_attentions=True)
    
    layer_atten  = outputs.attentions[attention_layer]
    layer_atten = layer_atten.squeeze(0)
    #
    sum_= layer_atten.sum(axis=1).sum(axis=0)
    sub_scores= sum_[1:-1].detach().cpu().numpy()
    #print(f"sub_scores:{sub_scores}")
    scores = []
    for key in keys:
        


        score = sum(sub_scores[key[0]:key[1]])/(key[1]-key[0])
        scores.append(score)
    assert len(words) == len(scores)
    sorted_ = sorted(enumerate(scores),key=lambda x:x[1], reverse=True)
    sorted_words_idxs = [i[0] for i in sorted_]
    candidates, cand_idx = valid_select(words,sorted_words_idxs,pos_vocabulary,stopwords)
    result_num = min(len(candidates),int(alpha*len(words)))
    return candidates[:result_num], cand_idx[:result_num]
    #        sorted_subs = sorted(zip(all_subs_indices,all_sub_scores),key=lambda x:x[1], reverse=True)

def get_importance_words_v2(
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



def get_scores(outputs,re_outputs):
    label=torch.argmax(outputs)
    label = int(label)
    confidence = torch.nn.functional.softmax(re_outputs)
    scores = [ float(outputs[label]-logit[label]) for logit in confidence]
    return scores

def create_mask(words,tokenizer,maxlength):
    
    all_input_ids  = []
    for i in range(len(words)):
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

def get_importance_words_v4(
    model,tokenizer, sentence, pos_vocabulary,stopwords,
    alpha=0.1, max_length=40
    ):
    device = torch.device('cuda')
    model.zero_grad()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False, max_length=max_length)
    #print("words:{}".format(words))
    # print("keys:{}".format(keys))
    assert len(words) == len(keys)
    input_words = ['[CLS]'] + sub_words + ['[SEP]']
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(input_words)])
    assert len(sub_words) == (len(input_ids[0])-2)
    outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids>0).to(device))
    # output_label = torch.argmax(outputs.logits)
    mask_copies_ids = create_mask(words,tokenizer,max_length)
    re_outputs = model(input_ids=mask_copies_ids.to(device), token_type_ids=None, \
                    attention_mask=(mask_copies_ids>0).to(device))
    #print(re_outputs)
    scores = get_scores(outputs.logits[0],re_outputs.logits)
    #scores=scores
    
    
    assert len(words) == len(scores)
    #print('------sorting------')
    # for k,value in zip(words,scores):
    #     print(f"({k},{value})")
    sorted_words = sorted(zip(words, scores),key=lambda x:x[1], reverse=True)
    #print(f"sorted_words:{sorted_words}")
    sorted_ = sorted(enumerate(scores),key=lambda x:x[1], reverse=True)
    sorted_words_idxs = [i[0] for i in sorted_]
    candidates, cand_idx = valid_select(words,sorted_words_idxs,pos_vocabulary,stopwords)
    #print("After valid:{}".format(candidates))
    result_num = max(min(len(candidates),int(alpha*len(words))),1)
    return candidates[:result_num], cand_idx[:result_num]
    #        sorted_subs = sorted(zip(all_subs_indices,all_sub_scores),key=lambda x:x[1]

# if __name__ == '__main__':
# #for attation leading important words generated
#     file_name = '/data/zhanghData/AttentionDefense/data/words_pos.dict'
#     with open(file_name, 'rb') as f:
#         pos_vocabulary = pickle.load(f)
#     stopwords = get_stopwords()
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#     config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
#     model.cuda()
#     load_model_path = '/data/zhanghData/AttentionDefense/save_models/sst2_bert/base/best.pt'
#     model.load_state_dict(torch.load(load_model_path))
    
#     file_path = '/data/zhanghData/AttentionDefense/save_results/attacks/sst2_bert/base/pwws_0.2_1000.csv'
#     atk_instances = read_adv_files(file_path, b_flag = False)

#     for i in range(0,12):
#         atk_count = 0
#         orig_impo_count = 0
#         perd_impo_count = 0
#         perd_recall_count = 0
#         orig_recall_count = 0
#         for instance in atk_instances:
#             #sentence = instance.perd_text
#             orig_text = instance.orig_text
#             perd_text = instance.perd_text
#             #print(sentence)
#             candidates, cand_idx = get_importance_words(model,tokenizer,orig_text,pos_vocabulary=pos_vocabulary, stopwords=stopwords, alpha=0.2,attention_layer=i )
#             atk_count += len(instance.atk_changes)
#             orig_impo_count += len(candidates)
#             for ch in instance.atk_changes:
#                 if ch[0] in candidates:
#                     orig_recall_count +=1
#             candidates, cand_idx = get_importance_words(model,tokenizer,perd_text,pos_vocabulary=pos_vocabulary, stopwords=stopwords, alpha=0.2,attention_layer=i )
#             perd_impo_count += len(candidates)
#             for ch in instance.atk_changes:
#                 if ch[1] in candidates:
#                     perd_recall_count +=1
#         print("-------- Layer {} --------".format(i))
#         print("recall: original {:.2f}%, attack {:.2f}%".\
#             format(100*(orig_recall_count/atk_count),100*(perd_recall_count/atk_count)))
#         print("accuracy: original {:.2f}%, attack {:.2f}%".\
#             format(100*(orig_recall_count/orig_impo_count),100*(perd_recall_count/perd_impo_count)))
