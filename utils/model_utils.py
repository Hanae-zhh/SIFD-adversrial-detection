from transformers import BertTokenizer, BertConfig 
from transformers import BertForSequenceClassification, AdamW 
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, DebertaForSequenceClassification
from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers import get_linear_schedule_with_warmup 
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import English
from utils.data_utils import load_pkl

def load_pre_models(base_model,num_labels,dropout):

    tokenizer, config, model = None, None, None  
    if base_model == 'roberta':
        model_type = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
        config = RobertaConfig.from_pretrained(model_type, num_labels=num_labels, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=0, hidden_dropout_prob=dropout,)
        model = RobertaForSequenceClassification.from_pretrained(model_type, config=config) 
    elif base_model == 'bert':
        model_type = 'bert-base-uncased' 
        tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = BertConfig.from_pretrained(model_type, num_labels=num_labels, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=0, hidden_dropout_prob=dropout,)
        model = BertForSequenceClassification.from_pretrained(model_type, config=config)
    elif base_model == 'deberta':
        model_type = 'microsoft/deberta-base' 
        tokenizer = DebertaTokenizer.from_pretrained(model_type, do_lower_case=True)
        # Load the pretrained BERT model
        config = DebertaConfig.from_pretrained(model_type, num_labels=num_labels, output_attentions=False, output_hidden_states=False, \
                        attention_probs_dropout_prob=0, hidden_dropout_prob=dropout,)
        model = DebertaForSequenceClassification.from_pretrained(model_type, config=config)
    model.cuda()
    return model,tokenizer



class NN_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN_model, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim  = output_dim

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class NN_config:
    def __init__(self,vocab_size,num_labels,max_length,word_to_idx) :
        self.vocab_size = vocab_size
        self.embed_size = 300
        self.num_feature_maps = 100
        self.stride = 1
        self.dropout_rate = 0.1
        self.filter_sizes = [2, 3, 4]
        self.num_class = num_labels
        self.hidden_size = 128
        self.num_layers = 1
        nlp = English()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.spacy_tokenizer = nlp.tokenizer
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss()
    
def load_word_to_id(args):
    pre_trained_base = "./data/pretrained/{}/{}".format(args.model_type,args.dataset)
    base_path = "{}/data".format(pre_trained_base)
    word_to_idx = load_pkl("{}/{}".format(base_path, "word_to_idx.pkl"))
    vocab = list(load_pkl("{}/{}".format(base_path, "vocab.pkl"))) 
    return word_to_idx, vocab