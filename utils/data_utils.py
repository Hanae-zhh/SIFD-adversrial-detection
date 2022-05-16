from cProfile import label
import pickle
import numpy as np
import logging
import sys
import torch
from torch.utils.data import Dataset
import re
import os
import random
class NN_Text(Dataset):
    def __init__(self, x , y):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx].astype('float32'))
        labels = torch.tensor(self.y[idx].astype('float32')).unsqueeze(0)
        return data, labels


def  load_count_vector_dict():
    file_name = '/data/zhanghData/AttentionDefense/data/counter-fitted-vectors.dict'
    with open(file_name, 'rb') as f:
            glove_ = pickle.load(f)
    return glove_

def cos_simlarity(golve_model,w1,w2):
    if w1 in golve_model:
        embed_w1 = golve_model[w1]
    else:
        embed_w1 = golve_model['UNK']
    if w2 in golve_model:
        embed_w2 = golve_model[w2]
    else:
        embed_w2 = golve_model['UNK']
    num = float(np.dot(embed_w1, embed_w2))  # 向量点乘
    denom = np.linalg.norm(embed_w1) * np.linalg.norm(embed_w2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def get_sst_data(type='train'):
    '''SST-2 GLUE version
    '''
    print('Getting Data...')
    texts = [] 
    labels = [] 
    #path = './data/sst2-mini/' + type + '.tsv'
    path = '/data/zhanghData/AttentionDefense/data/sst2/' + type + '.tsv'
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin.readlines()[1:]:
            line = line.strip().split('\t')
            texts.append(line[0])
            labels.append(1. if line[1]=='1' else 0.)
    print('Done, load {} datas from sst2 train dataset.'.format(len(texts)))
    return texts, labels 

def  get_pos_vocabulary():
    file_name = '/data/zhanghData/AttentionDefense/data/words_pos.dict'
    with open(file_name, 'rb') as f:
            pos_vocabulary = pickle.load(f)
    return pos_vocabulary



def get_stopwords():
    '''
    stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
    '''
    
    stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 
    'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 
    'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before',
     'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot',
      'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 
      'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 
      'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence', 
      'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 
      'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 
      'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 
      'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 
      'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 
      'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please', 's', 'same', 
      'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 
      'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 
      'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 
      'thru', 'thus', 'to', 'too', 'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', 
      "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 
      'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 
      'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you',
       "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '', ';', 'because', 'tv', "'s", '--', 
       'wo', 'some', '-', 'de', 'ca', 'so', "'ll", "'m", 'despite', 'two', 'should', 'might', "'d", 'inside', 'three', 'be', 'like',
        ')', '.', '...', '``', 'though', 'will', "'", 'each', "''", ',', 'since', 'every', '?', '(', ':', '`', 'us', 'go', '!', 'do',
        'I','So','"','|']
    return stop_words

def get_supported_pos_tags():
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
    return supported_pos_tags


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


def build_logging(logging_file_path, **kwargs):

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

def clean_str(string, tokenizer=None):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)
    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )

def cut_raw(seq, max_len):
    assert isinstance(seq, list)
    return seq[:max_len]


def save_pkl(file, path):
    with open(path, "wb") as handle:
        pickle.dump(file, handle)


def load_pkl(path):
    path = os.getcwd()+'/'+path
    with open(path, "rb") as handle:
        return pickle.load(handle)

def shuffle_lists(*args):
    """
    See https://stackoverflow.com/a/36695026
    """
    zipped = list(zip(*args))
    random.shuffle(zipped)
    return [list(x) for x in zip(*zipped)]

def pad(max_len, seq, token):
    assert isinstance(seq, list)
    abs_len = len(seq)

    if abs_len > max_len:
        seq = seq[:max_len]
    else:
        seq += [token] * (max_len - abs_len)

    return seq

def prep_seq(seq, word_to_idx, unk_token):
    assert isinstance(seq, list)
    seq_num = []

    for word in seq:
        try:
            seq_num.append(word_to_idx[word])
        except KeyError:
            seq_num.append(word_to_idx[unk_token])

    return seq_num