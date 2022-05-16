from ast import arg
from cProfile import label
from cgitb import text
from operator import imod
from textattack.datasets import huggingface_dataset
import numpy as np
import argparse 
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, \
                                        GeneticAlgorithmAlzantot2018, BERTAttackLi2020, DeepWordBugGao2018, \
                                        TextBuggerLi2018, PSOZang2020, BAEGarg2019, FasterGeneticAlgorithmJia2019
from textattack import AttackArgs
from textattack.datasets import Dataset
from textattack import Attacker
from textattack.loggers.attack_log_manager import AttackLogManager
# from pytorch_model import Model as pytorch_cnn
import textattack
# import dataloader
import os 
import sys 
import torch
import torch.nn.functional as F
# import models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertConfig 
from transformers import BertForSequenceClassification, AdamW
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification 
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from config import Args
from tqdm import tqdm
import csv
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle
from utils.model_utils import load_model
from utils.data_utils import pad,prep_seq,load_pkl,cut_raw,clean_str
from models.lstm import LSTM
from models.cnn import CNN
from spacy.lang.en import English
class CustomPytorchModelWrapper_bert(ModelWrapper):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        # BERT-Attack
        # self.device = torch.device('cuda:1')
        # TextFooler and TextBugger
        self.device = torch.device('cuda')
        self.max_length = 40 if args.dataset=='sst2' else 200


    def __call__(self, text_input_list):

        # text_input_list = [text.split(' ') for text in text_input_list]
        # prediction = self.model.text_pred(text_input_list)
        all_input_ids = self.encode_fn(self.tokenizer, text_input_list)
        dataset = TensorDataset(all_input_ids)
        pred_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        prediction = []
        for batch in pred_dataloader:
            outputs = self.model(batch[0].to(self.device), token_type_ids=None, attention_mask=(batch[0]>0).to(self.device))
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            prediction.append(logits)

        return np.concatenate(prediction, axis=0)

    def encode_fn(self, tokenizer, text_list):
        all_input_ids = []    
        for text in text_list:
            input_ids = self.tokenizer.encode(
                            text,
                            truncation=True,                       
                            add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                            max_length = self.max_length,           # 设定最大文本长度 200 for IMDb and 40 for sst
                            # pad_to_max_length = True,   # pad到最大的长度  
                            padding = 'max_length',
                            return_tensors = 'pt'       # 返回的类型为pytorch tensor
                       )
            all_input_ids.append(input_ids)    
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids

class CustomPytorchModelWrapper_nn(ModelWrapper):
    def __init__(self, model,word_to_idx, args):
        self.model = model
        self.word_to_idx = word_to_idx
        # BERT-Attack
        # self.device = torch.device('cuda:1')
        # TextFooler and TextBugger
        self.device = torch.device('cuda')
        self.max_length = args.max_length
        self.unk_token = args.unk_token
        self.pad_token = args.pad_token
        nlp = English()
        self.spacy_tokenizer = nlp.tokenizer
    def __call__(self, text_input_list):

        # text_input_list = [text.split(' ') for text in text_input_list]
        # prediction = self.model.text_pred(text_input_list)
        input_list = [ cut_raw(clean_str(text, tokenizer=self.spacy_tokenizer),self.max_length)  for text in text_input_list]
        inputs = [pad(self.max_length, x, self.pad_token) for x in input_list]
        inputs = torch.tensor(
            [prep_seq(x, self.word_to_idx, self.unk_token) for x in inputs],
            dtype=torch.int64,).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        #print("Output:{}".format(outputs.size))
        return outputs

def load_sst2_dataset(data_num,type='dev'): 
    print('Getting Data...')
    dataset  = []
    #path = './data/SST-2-mini/' + type + '.tsv'
    path = './data/sst2/' + type + '.tsv'
    count  = 0
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin.readlines()[1:]:
            count += 1
            line = line.strip().split('\t')
            text=line[0]
            label= 1 if line[1]=='1' else 0
            dataset.append((text,label))
            if count >= data_num:
                break
    dataset = textattack.datasets.Dataset(dataset,shuffle=False)
    print('Done: Total datas:{}'.format(len(dataset)))
    return dataset

def load_imdb_dataset(data_num=100):
    print("Geting imdb dataset...")
    dataset  = []
    doc_count = 0  # number of input sentences
    path = './data/imdb/dev_{}.csv'.format(data_num)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        text = line[0]
        label = 1 if line[1]=='1.0' else 0
        doc_count += 1
        dataset.append((text,label))
    dataset = textattack.datasets.Dataset(dataset)
    print('Done: Total datas:{}'.format(len(dataset)))
    return dataset

def load_agnews_dataset(data_num=100):
    print("Geting imdb dataset...")
    dataset  = []
    doc_count = 0  # number of input sentences
    path = './data/agnews/test_{}.csv'.format(data_num)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        text = line[0]
        label = int(line[1])
        doc_count += 1
        dataset.append((text,label))
    dataset = textattack.datasets.Dataset(dataset)
    print('Done: Total datas:{}'.format(len(dataset)))
    return dataset

def create_save_path(args):
    if args.fada:
        attacker_log_path = os.path.join("/data/zhanghData/AttentionDefense/save_results/attacks",\
        "{}_{}".format(args.dataset,args.model_type),"{}_fada".format(args.train_mode))
    else:
        attacker_log_path = os.path.join("/data/zhanghData/AttentionDefense/save_results/attacks",\
        "{}_{}".format(args.dataset,args.model_type),args.train_mode)
    
    if not os.path.exists(attacker_log_path):
        os.makedirs(attacker_log_path)
    
    txt_file = os.path.join(attacker_log_path, '{}_{}_{}.txt'.format(args.attack,args.modify_rate,args.attack_num))
    csv_file = os.path.join(attacker_log_path, '{}_{}_{}.csv'.format(args.attack,args.modify_rate,args.attack_num))
    return txt_file, csv_file


def load_word_to_id(args):
    base_path = "{}/data".format(args.pre_trained_base)
    word_to_idx = load_pkl("{}/{}".format(base_path, "word_to_idx.pkl"))
    vocab = list(load_pkl("{}/{}".format(base_path, "vocab.pkl"))) 
    return word_to_idx, vocab

def attack(args):
    # model_path = '/public1014/zhub/TextClassificationBert/model/freelb_orig/9.pt'
    # if args.base:
    #     model_path = '/public1014/zhub/TextDefender/saved_models/imdb_bert/base-len200-epo10-batch24-epoch9.pth'
    # BERT
    tokenizer, config, model = None, None, None 
    print(f"Loading model dict:{args.load_model_path}")
    if args.model_type in ['lstm','cnn']:
        word_to_idx, vocab = load_word_to_id(args)
        args.vocab_size = len(vocab)
        if args.model_type == "cnn":
            model = CNN(args)
        else:
            model = LSTM(args)
        model = load_model(args.load_model_path,model)
        model.cuda()
        model_wrapper = CustomPytorchModelWrapper_nn(model,word_to_idx,args)
    else:
        if args.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            config = BertConfig.from_pretrained('bert-base-uncased', num_labels=args.num_class, output_attentions=False, output_hidden_states=False)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

        # ROBERTA
        elif args.model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
            config = RobertaConfig.from_pretrained('roberta-base', num_labels=args.num_class, output_attentions=False, output_hidden_states=False)
            model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
        
        # DEBERTA
        elif args.model_type == 'deberta':
            
            tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', do_lower_case=True)
            config = DebertaConfig.from_pretrained('microsoft/deberta-base', num_labels=args.num_class, output_attentions=False, output_hidden_states=False)
            model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', config=config)
        model.load_state_dict(torch.load(args.load_model_path))
        model.cuda()
        model_wrapper = CustomPytorchModelWrapper_bert(model, tokenizer, args)
    # from FAD.tf_fada_sst import dataset as tf_sst_dataset 
    # dataset = textattack.datasets.Dataset(tf_sst_dataset)
    
    if args.dataset == 'sst2':
        #dataset = HuggingFaceDataset("glue", "sst2", "validation")
        dataset = load_sst2_dataset(args.attack_num,type = 'dev')

    elif args.dataset == 'imdb':
        #dataset = HuggingFaceDataset("imdb", None, "test")
        dataset = load_imdb_dataset(args.attack_num)
    elif args.dataset == 'agnews':
        #dataset = HuggingFaceDataset("ag_news", None, "test")
        dataset = load_agnews_dataset(args.attack_num)
  
    modify_rate = args.modify_rate 
    
    # pop = 2 # 60
    # iters = 2 # 20

    #attack = TextBuggerLi2018.build(model_wrapper, modify_rate )

    if args.attack == "textfooler":
        attack = TextFoolerJin2019.build(model_wrapper, modify_rate )
    elif args.attack == "bae":
        attack = BAEGarg2019.build(model_wrapper, modify_rate )
    elif args.attack == 'fastgenetic':
        attack = FasterGeneticAlgorithmJia2019.build(model_wrapper, modify_rate , 10, 10)
    elif args.attack == 'pso':
        attack = PSOZang2020.build(model_wrapper, modify_rate , 2, 2)
    elif args.attack == 'textbugger':
        attack = TextBuggerLi2018.build(model_wrapper, modify_rate )
    elif args.attack== 'deepwordbug':
        attack = DeepWordBugGao2018.build(model_wrapper,modify_rate)
    elif args.attack == 'pwws':
        attack = PWWSRen2019.build(model_wrapper,modify_rate)
    elif args.attack == 'bertattack':
        attack = BERTAttackLi2020.build(model_wrapper,modify_rate)
    # attack_args = AttackArgs(num_examples=26940, checkpoint_dir="checkpoints", shuffle=True, log_to_csv='./FAD', csv_coloring_style='plain')
    
    txt_file,csv_file = create_save_path(args)
    attack_args = AttackArgs(num_examples=args.attack_num,log_to_txt=txt_file,log_to_csv=csv_file,checkpoint_dir="checkpoints", shuffle=True)
    #attack_args = AttackArgs(num_examples=args.attack_num,checkpoint_dir="checkpoints", shuffle=True)
    attacker = Attacker(attack, dataset, attack_args)

    start_time = time.time()
    attacker.attack_dataset()
    end_time = time.time()
    print(f"-------Attack time:{(end_time-start_time)/60} min")

if __name__ == '__main__':
    args = Args()
    print(args)

    attack(args)
#model_path:指定模型
#model_dir:指定训练模型存放的文件夹
#CUDA_VISIBLE_DEVICES=0 python3 run_attack.py  --dataset sst2 --modify_rate 0.2 --attack textfooler --attack_num 100 --model_path sst2_bert/base/best.pt --train_mode base
#
#CUDA_VISIBLE_DEVICES=1 python3 run_attack.py  --dataset snli --modify_rate 0.0 --attack bae --attack_num 1000  --model_type roberta --train_mode base --model_path /data/zhangh/GAT/save_models/snli_roberta/standard-len128-epo10-batch16-best.pth

#CUDA_VISIBLE_DEVICES=2 python3 run_attack.py  --dataset snli --modify_rate 0.1 --attack textfooler --attack_num 5000  --model_type bert

# CUDA_VISIBLE_DEVICES=2 python3 run_attack.py  --dataset snli --modify_rate 0.0 --attack pwws --attack_num 1000  --model_type roberta --train_mode base --model_path /data/zhangh/GAT/save_models/snli_roberta/standard-len128-epo10-batch16-best.pth

    
#Attack Shell
# CUDA_VISIBLE_DEVICES=1 python3 run_attack.py  --dataset imdb --modify_rate 0.1 --attack bae --attack_num 3000  --train_mode base --model_type bert
# CUDA_VISIBLE_DEVICES=1 python3 run_attack.py  --dataset imdb --modify_rate 0.1 --attack deepwordbug --attack_num 3000  --train_mode base --model_type bert
# CUDA_VISIBLE_DEVICES=1 python3 run_attack.py  --dataset imdb --modify_rate 0.1 --attack deepwordbug --attack_num 1000  --train_mode base --model_type bert
# CUDA_VISIBLE_DEVICES=2 python3 run_attack.py  --dataset imdb --modify_rate 0.1 --attack pwws --attack_num 1000  --train_mode base --model_type bert

# CUDA_VISIBLE_DEVICES=0 python3 run_attack.py  --dataset agnews --modify_rate 0.15 --attack pwws --attack_num 7600  --train_mode base --model_type bert
# CUDA_VISIBLE_DEVICES=0 python3 run_attack.py  --dataset agnews --modify_rate 0.2 --attack bae --attack_num 7600  --train_mode base --model_type bert

# CUDA_VISIBLE_DEVICES=0 python3 run_attack.py  --dataset agnews --modify_rate 0.2 --attack pwws --attack_num 100  --train_mode base --model_type lstm