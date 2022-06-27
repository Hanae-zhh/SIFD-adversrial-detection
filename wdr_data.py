
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
import time
from utils.attack_instance import read_adv_files
from utils.data_utils import pre_tokenize
from utils.data_utils import get_pos_vocabulary, get_stopwords
from utils.attack_instance import read_adv_files
from utils.sensitivity import valid_select, create_mask
from utils.data_utils import pad,prep_seq,load_pkl,cut_raw,clean_str
from models.lstm import LSTM
from models.cnn import CNN
from spacy.lang.en import English
from utils.model_utils import NN_config,load_word_to_id,load_model
def get_scores(output,re_outputs):
    scores = []
    label=torch.argmax(output)
    label=int(label)
    # re_labels = torch.argmax(re_outputs,dim=1)
    # re_labels = re_labels.detach().cpu().numpy()
    # # print("---------In get scores---------")
    # # print(f'label:{label}')
    # # print(f"re_labels:{re_labels}")
    # confidence = torch.nn.functional.softmax(re_outputs,dim=1)
    # #print(confidence)
    for x in re_outputs:
        logit = x.detach().cpu().numpy()
        logit_delete = np.delete(logit,label)
        scores.append(float(logit[label]-np.max(logit_delete)))

    return label, scores
def create_unk(input_ids, cand_ids):
    mask_inputs_list = []
    for idx in cand_ids:
        inputs = copy.deepcopy(input_ids)
        inputs[idx] = 0
        mask_inputs_list.append(inputs)
    return mask_inputs_list
def sentence_sensitivity_base_information(
    model,tokenizer, sentence,pos_vocabulary,stopwords,
    max_length,batch_size
    ):
    device = torch.device('cuda')
    model.eval()
    words,sub_words,keys = pre_tokenize(sentence, tokenizer, use_MASK=False)
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
    
    candidates, cand_ids = words,[i for i in range(len(words))]
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

    label, scores = get_scores(outputs.logits[0],logits)
    if len(scores) < 512:
        scores += [0]*(512-len(scores))
    else:
        scores = scores[:512]
    assert len(scores) == 512
    return label, scores

def sentence_sensitivity_base_information_nn(model,nn_config, sentence):
    device = torch.device('cuda')
    model.eval()
    input_words = cut_raw(clean_str(sentence, tokenizer=nn_config.spacy_tokenizer),nn_config.max_length)
    input_ = pad(nn_config.max_length, input_words, nn_config.pad_token) 
    input_ids = torch.tensor([prep_seq(input_, nn_config.word_to_idx, nn_config.unk_token)],
                                dtype=torch.int64,).to(device)
    input_embed = model.embedding(input_ids)
    outputs = model(inputs=None,embeddings=input_embed)

    mask_copies_ids = create_unk(input_ids[0].detach().cpu().numpy(), [i for i in range(len(input_words))])
    mask_inputs = torch.tensor(mask_copies_ids,dtype=torch.int64,).to(device)
    with torch.no_grad():
        mask_input_embed = model.embedding(mask_inputs)
        mask_outputs = model(inputs=None,embeddings=mask_input_embed)

    label, scores = get_scores(outputs[0],mask_outputs)
    if len(scores) < 512:
        scores += [0]*(512-len(scores))
    else:
        scores = scores[:512]
    assert len(scores) == 512
    return label, scores


def get_word_scores(sub_scores,cand_ids,keys):
    word_scores = []
    for i in range(len(keys)):
        if i in cand_ids:
            if keys[i][0]+1 == keys[i][1]:
                word_scores.append(sub_scores[keys[i][0]])
            else:
                word_scores.append(max(sub_scores[keys[i][0]:keys[i][1]]))
    return word_scores

if __name__ == '__main__':

    ''''''
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str,default="sst2")
    argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
    argparser.add_argument("--model_type", type=str, default="bert")
    args = argparser.parse_args()
    device = torch.device('cuda')
   
    max_length = 40 if args.dataset =='sst2' else 200 if args.dataset =='imdb' else 50
    num_labels = 4 if args.dataset == 'agnews' else 2
    batch_size = 64 if args.dataset =='sst2' else 16 if args.dataset =='imdb' else 50
    
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels, output_attentions=False, output_hidden_states=False)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        model.cuda()
        model.load_state_dict(torch.load(load_model_path))

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
    count = 0 
    start_time = time.time()
    attack_names = args.atk_path.split("_")
    for atk in atk_instances:
        count += 1
        print("process information----count{}/{}".format(count,attack_names[2]))
        # orig_softmax,pre_label,softmaxs,scores,re_labels=sentence_sensitivity_base_information(\
        #         model,tokenizer, atk.orig_text,pos_vocabulary,stopwords,max_length,batch_size)
        if args.model_type in ['lstm','cnn']:
            pre_label, orig_scores = sentence_sensitivity_base_information_nn(\
                model,nn_config, atk.orig_text)
            perd_pre_label, perd_scores = sentence_sensitivity_base_information_nn(\
                model,nn_config, atk.perd_text)
        #print(scores,pre_label,re_labels)
        else:
            pre_label, orig_scores = sentence_sensitivity_base_information(\
                    model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
            perd_pre_label, perd_scores = sentence_sensitivity_base_information(\
                    model,tokenizer, atk.perd_text,pos_vocabulary,stopwords,max_length,batch_size)
        if pre_label != atk.ground or perd_pre_label == atk.ground:
            continue
        normal_all_info.append(orig_scores)
        attack_all_info.append(perd_scores)
    end_time = time.time()
    dir_path = '/data/zhanghData/AttentionDefense/data/wdr_detector_data'
    save_folder = os.path.join(dir_path,"{}_{}".format(args.dataset,args.model_type))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    info_length = len(normal_all_info)
    normal_file_path = os.path.join(save_folder,"normal_{}_{}_{}.npy".format(attack_names[0],attack_names[1],info_length))
    attack_file_path = os.path.join(save_folder,"attack_{}_{}_{}.npy".format(attack_names[0],attack_names[1],info_length))
    np.save(normal_file_path,normal_all_info)
    np.save(attack_file_path,attack_all_info)
    print("save total info:{} !".format(info_length))
    print("Time Consumption:{:.2f}min".format((end_time-start_time)/60))

    # CUDA_VISIBLE_DEVICES=1 python3 wdr_data.py --dataset agnews --model_type cnn --atk_path textfooler_0.2_7600 
    # CUDA_VISIBLE_DEVICES=2 python3 wdr_data.py --dataset agnews --model_type cnn --atk_path deepwordbug_0.2_7600
    # CUDA_VISIBLE_DEVICES=2 python3 wdr_data.py --dataset agnews --model_type cnn --atk_path pwws_0.2_7600
    # CUDA_VISIBLE_DEVICES=0 python3 wdr_data.py --dataset agnews --model_type cnn --atk_path bae_0.2_7600

    # CUDA_VISIBLE_DEVICES=1 python3 wdr_data.py --dataset imdb --model_type cnn --atk_path pwws_0.1_5000
    # CUDA_VISIBLE_DEVICES=1 python3 wdr_data.py --dataset imdb --model_type cnn --atk_path bae_0.1_5000 
    # CUDA_VISIBLE_DEVICES=0 python3 wdr_data.py --dataset imdb --model_type cnn --atk_path textfooler_0.1_5000 
    # CUDA_VISIBLE_DEVICES=0 python3 wdr_data.py --dataset imdb --model_type cnn --atk_path deepwordbug_0.1_5000
