#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/12/03 14:57:18
@Author      :zhangh
@version      :1.0
'''
import argparse
from email.policy import default
import os
import sys

class Args:
     
    #parser = argparse.ArgumentParser(description="args for experiments")
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str, default='sst2')
    argparser.add_argument("--model_path", type=str, default="")
    argparser.add_argument("--model_type", type=str, default='bert')
    argparser.add_argument("--mode", type=str, default='train') #['train','test','attack']
    
    #for attack
    argparser.add_argument("--attack_num", type=int, default=1000)
    #argparser.add_argument("--at-set", type=str, default="")
    argparser.add_argument("--attack", type=str, default="textbugger")  
    argparser.add_argument("--pop", type=int, default=2)
    argparser.add_argument("--iters", type=int, default=2)
    argparser.add_argument("--modify_rate", type=float, default=0.0)
    argparser.add_argument("--model_dir", type=str, default="")

    #for train
    argparser.add_argument("--fada", action='store_true', help="whether to use fada")
    argparser.add_argument("--ada", action='store_true', help="whether to use ada")
    argparser.add_argument("--train_mode",type=str,default='base')
    argparser.add_argument('--aug_only', action='store_true')
    argparser.add_argument("--epochs",type=int,default=10)
    argparser.add_argument("--adv_K", type=int, default=3)
    argparser.add_argument("--train_lr", type=float, default=2e-5)
    argparser.add_argument("--dropout", type=float, default=0.1)
    

    #for defense 
    #argparser.add_argument("--atk_path", type=str, default="",help="The attack result--file name")
    argparser.add_argument("--top_k",type=int,default=3)
    argparser.add_argument("--alpha",type=float,default=0.2)
    argparser.add_argument("--use_mask",action='store_true', help="Whether use [MASK] when infactoring")
    argparser.add_argument("--threshold", type=float, default=0.9)
    argparser.add_argument("--vote_num",type=int,default=1)


    args_ = argparser.parse_args()

    #模型保存路径
    #save_model_dir ="/data/zhanghData/AttentionDefense/save_models"
    save_model_dir ="/data/zhanghData/AttentionDefense/save_models"
    dataset = args_.dataset
    model_type =args_.model_type
    if dataset in ['imdb','sst2']:
        num_class = 2
    elif dataset == 'agnews':
        num_class = 4
    elif dataset == 'snli':
        num_class = 3
    mode =args_.mode
    #model_path = args_.model_path#[bert,roberta,deberta]
    
    
    #for train
    fada_path='./data/friendly-tf-sst-2021-10-28-17-42-log.csv'
    ada_path='./data/atk_tf_sst.csv'
    train_mode = args_.train_mode # [base, freelb, fgm]
    fada = args_.fada
    ada = args_.ada
    aug_only = args_.aug_only
    epochs = args_.epochs
    hidden_dropout_prob  =args_.dropout
    train_lr  = args_.train_lr
    
    #cnn lstm
    glove_path = "/data/zhanghData/AttentionDefense/data/pretrained/gloVe/glove.42B.300d.txt"
    seed_val = 42
    path_to_imdb = '/data/zhanghData/Datasets/aclImdb'
    path_to_agnews = '/data/zhanghData/Datasets/ag_news_csv'
    val_split_size = 1000
    pad_token = "<pad>"
    eos_token = "."
    unk_token = "<unk>"
    pre_trained_base = "./data/pretrained/{}/{}".format(model_type,dataset)
    path_to_pre_trained_init = os.path.join(pre_trained_base,"pretrained_init.npy")
    path_to_dist_mat = os.path.join(pre_trained_base,"dist_mat.npy")
    path_to_idx_to_dist_mat_idx_dict = os.path.join(pre_trained_base,"idx_to_dist_mat_idx.pkl")
    path_to_dist_mat_idx_to_idx_dict = os.path.join(pre_trained_base,"dist_mat_idx_to_idx.pkl")
    if model_type in ['lstm','cnn']:
        epochs = 20
    
    # Training params
    vocab_size = 0
    embed_size = 300
    dropout_rate = 0.1
    # Params for CNN
    filter_sizes = [2, 3, 4]
    stride = 1
    num_feature_maps = 100

    # Params for LSTM
    hidden_size = 128
    num_layers = 1

    #for attack
    attack_num = args_.attack_num
    attack = args_.attack
    modify_rate = args_.modify_rate

    #for defense 
    top_k=args_.top_k
    alpha=args_.alpha
    use_mask= args_.use_mask
    threshold = args_.threshold
    vote_num = args_.vote_num

    if args_.model_path != "":
        load_model_path = os.path.join(save_model_dir,args_.model_path)
    else:
        if args_.fada:
            load_model_path = os.path.join(save_model_dir,\
            "{}_{}".format(dataset, model_type),\
            "{}_fada".format(args_.train_mode),"best.pt")
        else:
            load_model_path = os.path.join(save_model_dir,\
                "{}_{}".format(dataset, model_type),\
                args_.train_mode,"best.pt")
    #load_model_path = "/data/ZhanghData/MaskDefense/save_models/imdb_bert/standard-len256-epo10-batch16-best.pth"
   
    ## 建立模型保存的路径，文件夹
    if args_.model_dir != "":
        save_path = os.path.join(save_model_dir,args_.model_dir)
    else:
        if fada:
            save_path = os.path.join(save_model_dir, "{}_{}".format(dataset, model_type),"{}_fada".format(train_mode))
        elif ada:
            save_path = os.path.join(save_model_dir, "{}_{}".format(dataset, model_type),"{}_ada_{}".format(train_mode, attack))
        else:
            save_path = os.path.join(save_model_dir, "{}_{}".format(dataset, model_type),train_mode)
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except:
        print("Error: Model override warning!")

    if model_type in ['lstm','cnn']:
        batch_size = 256
    else:
        if dataset == 'sst2':
            batch_size = 64  
        elif dataset=="imdb":
            batch_size = 24
        elif dataset=="agnews":
            batch_size = 64
    
    if dataset =='sst2':
        max_length = 40 
    elif dataset == 'imdb':
        max_length = 200
    elif dataset == 'agnews':
        max_length = 50

    fgm_epsilon = 1.0
    adv_lr = 1e-2
    adv_K = 3 #"should be at least 1"
    adv_init_mag=2e-2
    adv_norm_type = "l2" #choices=["l2", "linf"]
    adv_max_norm = 0 # set to 0 to be unlimited
    
    
    attention_probs_dropout_prob = 0
    gradient_accumulation_steps = 1
    pgd_alpha = 0.3
    pgd_epsilon = 1.0
    pgd_k = 3

 

    