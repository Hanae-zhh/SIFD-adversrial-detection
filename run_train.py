from ast import arg
from cProfile import label
from http.client import ImproperConnectionState
import torch 
import random 
import numpy as np 
import time 
import os 
import csv 
import argparse 
import sys 
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader, random_split 
from transformers import BertTokenizer, BertConfig 
from transformers import BertForSequenceClassification, AdamW 
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup 
from adversarial_train import FreeLB, PGD, FGM 
from config import Args
import logging
import torch.optim as optim
from models.cnn import CNN
from models.lstm import LSTM
import torch.nn as nn
from utils.data_module import DataModule
import math
from utils.data_utils import shuffle_lists,pad,prep_seq
from utils.attack_instance import read_adv_files
def build_trainlog(args, **kwargs):

    logging_file_path  = os.path.join(args.save_path, \
        '{}_{}.log'.format(args.mode,args.dataset))
    
    logger = logging.getLogger('Huanlogger')
    logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler(sys.stderr)#默认是sys.stderr
    rf_handler.setLevel(logging.DEBUG) 
    #rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    rf_handler.setFormatter(logging.Formatter("%(asctime)s --%(name)s-- %(message)s"))

    f_handler = logging.FileHandler(logging_file_path)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def flat_accuracy(preds, labels):
    
    """A function for calculating accuracy scores"""
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc = sum(int(t) for t in pred_flat==labels_flat) / len(pred_flat)
    # return accuracy_score(labels_flat, pred_flat)
    return acc 
# RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

def get_sst_data(args,type='train'):
    '''SST-2 GLUE version
    '''
    print('Getting Data...')
    texts = [] 
    labels = [] 
    #path = './data/sst2-mini/' + type + '.tsv'
    path = os.path.join(os.getcwd(),"datasets","sst2","{}.tsv".format(type))
    count = 0
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin.readlines()[1:]:
            line = line.strip().split('\t')
            texts.append(line[0])
            labels.append(1. if line[1]=='1' else 0.)
            count +=1
            # if count > 10000:
            #     break
    print('Done, load {} datas from sst2 train dataset.'.format(len(texts)))
    return texts, labels 

def get_imdb_data(args,type= 'train'):
    texts = []
    labels = []
    path = os.path.join(os.getcwd(),"datasets","imdb","{}.txt".format(type))
    with open(path, 'r', encoding='utf-8') as fin:
        count = 0
        for line in fin:
            data = line.strip().split('\t')
            texts.append(data[0])
            labels.append(1 if data[1]=='1' else 0)
            count += 1
    print('Done, load {} datas from imdb train dataset.'.format(len(texts)))
    return texts,labels

def get_agnews_files(type):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    #path = r'/data/zhanghData/Datasets/ag_news_csv/{}.csv'.format(type)
    path = os.path.join(os.getcwd(),"datasets","ag_news_csv","{}.csv".format(type))
    logger.info(f"load dateset from: {path}")
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        #content = line[1] + ". " + line[2]
        content = line[2] # delete title
        texts.append(content)
        labels_index.append(int(line[0])-1)
        #  Original labels are [1, 2, 3, 4] ->
        #                      ['World', 'Sports', 'Business', 'Sci/Tech']
        # Re-map to [0, 1, 2, 3].
        doc_count += 1
    return texts,labels_index

def get_yelp_data(type):
    texts = [] 
    labels = [] 
    t = 0
    path = os.path.join(os.getcwd(),"dataset","yelp","{}.csv".format(type))
    with open(path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        for line in reader:
            if line[1]=='label':
                continue
            texts.append(line[0])
            labels.append(1 if line[1]=='2' else 0)
    return texts,labels

def get_data(args):
    if args.dataset == 'sst2':
        train_texts, train_labels = get_sst_data(args,type='train')
    elif args.dataset == 'imdb':
        train_texts, train_labels = get_imdb_data(args,type='train')
    elif args.dataset == 'agnews':
        train_texts, train_labels =get_agnews_files(type='train')
    elif args.dataset == 'yelp':
        train_texts, train_labels =get_yelp_data(type='train')
    # # GLUE 版本没有测试集标签
    # test_texts, test_labels = get_data('dev')
    
    if args.aug_only:
        train_texts, train_labels = [], [] 
    if args.fada:
        print('Adding fada...')
        with open(args.fada_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            reader = list(reader)[1:]
            for line in reader:
                train_texts.append(str(line[7]))
                train_labels.append(int(float(line[5])))
    
    elif args.ada:
        with open(args.ada_path, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)
            reader = list(reader)[1:]
            for line in reader:
                if line[8] == 'Successful':
                    train_texts.append(str(line[7]))
                    train_labels.append(int(float(line[0])))
    
    return train_texts,train_labels

def encode_fn(tokenizer, text_list):
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,
                        truncation=True,                       
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = args.max_length,           # 设定最大文本长度
                        # pad_to_max_length = True,   # pad到最大的长度  
                        padding = 'max_length',
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

def build_inputs(batch):
    '''
    Sent all model inputs to the appropriate device (GPU on CPU)
    rreturn:
     The inputs are in a dictionary format
    '''
    input_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    batch = (batch[0].to(device), (batch[0]>0).to(device), None, batch[1].long().to(device))
  
    inputs = {key: value for key, value in zip(input_keys, batch)}
    return inputs

def make_dataloader(texts,labels,word_to_idx):
    sentences = [pad(args.max_length, sentence, args.pad_token)
            for sentence in texts]
    inputs_ids = [
        prep_seq(sentence, word_to_idx, args.unk_token)
        for sentence in sentences]
    inputs_ids = torch.tensor(inputs_ids)
    labels = torch.tensor(labels)
    dataset = TensorDataset(inputs_ids,labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    return dataloader

def inference(
    inputs,
    model,
    word_to_idx,
    config,
    val=False,
    single=False,
):
    softmax = nn.Softmax(dim=1) #dim=1代表在1维上sum=1
    model.eval()

    if single:
        assert isinstance(inputs, str)
        inputs = [inputs]
    else:
        assert isinstance(inputs, list)

    inputs = [pad(config.max_length, x, config.pad_token) for x in inputs]
    inputs = torch.tensor(
        [prep_seq(x, word_to_idx, config.unk_token) for x in inputs],
        dtype=torch.int64,
    ).to(device)

    with torch.no_grad():
        outputs = model(inputs)
    outputs = softmax(outputs)
    probs = outputs.cpu().detach().numpy().tolist()
    _, preds = torch.max(outputs, 1)
    #_最大值，preds最大值的index,1代表输出行最大值
    preds = preds.cpu().detach().numpy().tolist()
    '''
        detach(): 返回一个新的Tensor，但返回的结果是没有梯度的。
        cpu():把gpu上的数据转到cpu上。
        numpy():将tensor格式转为numpy。
    '''
    if single:
        preds, probs = preds[0], probs[0]
    if val:
        return preds, outputs
    else:
        return preds, probs

def compute_accuracy(preds, labels):
    assert len(preds) == len(labels)
    return len([True for p, t in zip(preds, labels) if p == t]) / len(preds)

def eval_nn_model(args,model,word_to_idx,eval_texts,eval_labels,criterion):
    # global best_epoch
    batch_size = args.batch_size
    num_batches = int(math.ceil(len(eval_texts) / batch_size))
    predictions = []
    total_loss = []
    for batch in range(num_batches):
        sentences = eval_texts[
            batch * batch_size : (batch + 1) * batch_size
        ]
        labels =eval_labels[
            batch * batch_size : (batch + 1) * batch_size
        ]
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
      
        preds, outputs = inference(
            sentences,
            model,
            word_to_idx,
            args,
            val=True,
        )

        predictions += preds
        loss = criterion(outputs, labels)
        total_loss.append(loss.item())

    acc = compute_accuracy(predictions,eval_labels)
    total_loss = np.mean(total_loss)

  
    return total_loss,acc
    # logger.log.info("Best epoch up to now: {}".format(best_epoch))
    # logger.log.info(
    #     "Val: epoch {}, loss {}, accuracy {}".format(epoch, total_loss, acc)
    # )


def run_train_nn(args):
    learning_rate = 1e-3
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()
    data_module = DataModule(args, logger, gpu=True)
    args.vocab_size = len(data_module.vocab)
    best_epoch = (0, np.inf)
    save_model_path = args.save_path
    inti_pretained = data_module.init_pretrained
    
    if args.model_type == 'cnn':
        model = CNN(args,logger,inti_pretained)
    else:
        model = LSTM(args,logger,inti_pretained)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1,args.epochs+1):
        data_module.train_texts, data_module.train_pols = shuffle_lists(
        data_module.train_texts, data_module.train_pols)
      
        total_loss = []
        model.train()
        num_batches = int(math.ceil(len(data_module.train_texts) / batch_size))
        for batch in range(num_batches):
            texts = data_module.train_texts[batch * args.batch_size : (batch + 1) * batch_size]
            labels = data_module.train_pols[batch * batch_size : (batch + 1) * batch_size]
            sentences = [
                pad(args.max_length, sentence, args.pad_token)
                for sentence in texts]
            inputs = [
                prep_seq(sentence, data_module.word_to_idx, args.unk_token)
                for sentence in sentences]

            inputs = torch.tensor(inputs, dtype=torch.int64).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device)

            model.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            print(
                "Epoch {}, batch {}/{}: loss {}".format(
                    epoch, batch, num_batches, loss.item()))
        logger.info("Train: epoch {}, loss {}".format(epoch, np.mean(total_loss)))
        
 
        eval_loss,eval_acc  = eval_nn_model(args,model,data_module.word_to_idx,\
                            data_module.val_texts,data_module.val_pols,criterion)
        # if eval_loss < best_epoch[1]:
        #     best_epoch = (epoch, total_loss)
        if eval_loss < best_epoch[1]:
            best_epoch = (epoch, eval_loss)
        
        logger.info("Best epoch up to now: {}".format(best_epoch))
        logger.info(
            "Val: epoch {}, loss {}, accuracy {}".format(epoch, eval_loss, eval_acc)
        )
        logger.info("Save model at epoch {}".format(epoch))
        save_path = os.path.join(save_model_path,"{}.pt".format(str(epoch)))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },save_path)

    os.rename(
        "{}/{}.pt".format(save_model_path, best_epoch[0]),
        "{}/best.pt".format(save_model_path),
    )


def run_train_bert(args):

    epochs = args.epochs #default 10
    batch_size = args.batch_size 
    save_path = args.save_path

    tokenizer, config, model = None, None, None  
    if args.model_type == 'roberta':
        model_type = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        config = RobertaConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = RobertaForSequenceClassification.from_pretrained(model_type, config=config) 
    elif args.model_type == 'bert':
        model_type = 'bert-base-uncased' 
        tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = BertConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = BertForSequenceClassification.from_pretrained(model_type, config=config)
    elif args.model_type == 'deberta':
        model_type = 'microsoft/deberta-base' 
        tokenizer = DebertaTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = DebertaConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = DebertaForSequenceClassification.from_pretrained(model_type, config=config)

    model.cuda()
    train_texts,train_labels = get_data(args)    
    print('Done, load total {} datas.'.format(len(train_texts)))
    logger.info('Done, load total {} datas.'.format(len(train_texts)))
    print('Encoding Data...')
    all_train_ids = encode_fn(tokenizer, train_texts)
    labels = torch.tensor(train_labels)
    print('Done...')
    # Split data into train and validation
    dataset = TensorDataset(all_train_ids, labels)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    #CELoss = torch.nn.CrossEntropyLoss()

    best_epoch = 0
    best_val = 0.0
    
    print('Start Trainging...')
    print(f"save_path:{save_path}")
    for epoch in range(epochs):
        logger.info(f"Epoch:{epoch}")
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            
            model.zero_grad()
            inputs = build_inputs(batch)

            outputs = model(**inputs)
            loss, logits = outputs[:2]
            loss.backward()
            # loss = CELoss(logits, batch[1].long().to(device))
            
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() 
            scheduler.step()

            if (step+1) % 100 == 0:
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                training_acc = flat_accuracy(logits, label_ids)
                print('\033[0;31;40m{}\033[0m,step:{},training_loss:{},training_acc:{}'.format(time.asctime(time.localtime(time.time())), step, loss.item(), training_acc))
                
        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                inputs = build_inputs(batch)
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                # loss = CELoss(logits, batch[1].long().to(device))    
                total_val_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        logger.info("avg_train_loss: {}, avg_val_loss: {}, avg_val_accuracy: {}".\
            format(avg_train_loss, avg_val_loss,avg_val_accuracy))
                
        if best_val < avg_val_accuracy:
            best_val = avg_val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path,"best.pt"))
        
        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Val Accuracy: {avg_val_accuracy}')
        print(f'Best Val Accuracy: {best_val}')
        print('Best Epoch:', best_epoch)
    
        print('Save model...')
        torch.save(model.state_dict(), os.path.join(save_path,str(epoch)+'.pt'))
        print('Done...')
    print("Training Finishing!")
    logger.info("---Final  Result---\n best_val: {} ,best_epoch: {} ".format(best_val,best_epoch))


def train(args):
    if args.model_type in ['lstm','cnn']:
        run_train_nn(args)
    else:
        run_train_bert(args)

def test(args,test_texts,test_labels):

    model_dir = args.save_path
    model_type = None
    
    print("Load agnews model from dir: {}".format(model_dir))
    logger.info("Load agnews model from dir: {}".format(model_dir))
    tokenizer, config, model = None, None, None  
    if args.model_type == 'roberta':
        model_type = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        config = RobertaConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = RobertaForSequenceClassification.from_pretrained(model_type, config=config) 
    elif args.model_type == 'bert':
        model_type = 'bert-base-uncased' 
        tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = BertConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = BertForSequenceClassification.from_pretrained(model_type, config=config)
    elif args.model_type == 'deberta':
        model_type = 'microsoft/deberta-base' 
        tokenizer = DebertaTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = DebertaConfig.from_pretrained(model_type, num_labels=args.num_class, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob, hidden_dropout_prob=args.hidden_dropout_prob,)
        model = DebertaForSequenceClassification.from_pretrained(model_type, config=config)

    #test_texts,test_labels = read_agnews_mini_files('test')
    
    logger.info("load [{}] datas.".format(len(test_texts)))
    all_test_ids = encode_fn(tokenizer, test_texts)
    test_labels = torch.tensor(test_labels)
    pred_data = TensorDataset(all_test_ids, test_labels)
    pred_dataloader = DataLoader(pred_data, batch_size=args.batch_size, shuffle=False)

    model.cuda()
    files =os.listdir(model_dir)
    logger.info('----test info----')
    best_model_path= ''
    best_test_acc = 0.0
    for file_name in files:
        if '.pt' in file_name:
            load_model_path = os.path.join(model_dir,file_name)
            model.load_state_dict(torch.load(load_model_path))
            
            model.eval()
            total_test_accuracy = 0
            for i, batch in tqdm(enumerate(pred_dataloader)):
                with torch.no_grad():
                    outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
                    logits = outputs[0]
                    
                    logits = logits.detach().cpu().numpy()
                    label_ids = batch[1].to('cpu').numpy()
                    total_test_accuracy += flat_accuracy(logits, label_ids)
            avg_test_accuracy = total_test_accuracy / len(pred_dataloader)
            if avg_test_accuracy  > best_test_acc:
                best_test_acc = avg_test_accuracy
                best_model_path = load_model_path
            #print('model: {load_model_path}; test_acc:{}'.format(avg_test_accuracy))
            logger.info('model: {}; test_acc:{}'.format(load_model_path,avg_test_accuracy))
            logger.info('best model path:{}\nbest test accuracy:{}'.format(best_model_path,best_test_acc))

if __name__ == '__main__':

# CUDA_VISIBLE_DEVICES=5 python3 run_train.py --dataset yelp --mode train --model_type bert
# CUDA_VISIBLE_DEVICES=1 python3 run_train.py --dataset agnews --mode test --model_type bert 
# CUDA_VISIBLE_DEVICES=1 python3 run_train.py --dataset sst2 --mode train --model_type lstm 
    args = Args()
    logger = build_trainlog(args)
    logger.info("---args info---")
    logger.info(f"batch_size:{args.batch_size}, epochs:{args.epochs}")
   
    set_seed(2021)
    device = torch.device('cuda')
    #print(args)
    s_time = time.time()
    if args.mode =='train':
        train(args)
    elif args.mode =='test':
        #read text from attack files and test the NEs
        atk_path = os.path.join(os.getcwd(),"save_results","attacks",'imdb_bert','bae_0.1_100.csv')
        atk_instances = read_adv_files(atk_path)
        test_texts = [atk.orig_text for atk in atk_instances]
        test_labels = [atk.ground for atk in atk_instances]
        test(args,test_texts,test_labels)
    e_time = time.time()
    logger.info("Training time:{:.2f}min".format((e_time-s_time)/60))
