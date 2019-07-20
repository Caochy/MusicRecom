import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding

class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        
        self.article_feature_len = config.getint('model', 'user_article_len')
        self.moments_lda_len = config.getint('model', 'moments_lda_len')
        self.hidden = config.getint('model', 'hidden_size')
        
        self.emb_size = config.getint('model', 'emb_size')
        # self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.feature = nn.Linear(self.article_feature_len, self.emb_size)
        self.moments_lda = nn.Linear(self.moments_lda_len, self.emb_size)
        
        self.UserEmb = nn.Embedding(190662, self.emb_size)
        self.age = nn.Embedding(6, self.emb_size)
        self.gender = nn.Embedding(2, self.emb_size)
        

        feature_size = self.emb_size * 3
        self.out = nn.Linear(feature_size, 2 * self.hidden)

        self.init_emb(self.UserEmb)
        self.init_emb(self.age)
        self.init_emb(self.gender)

    
    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])
        nn.init.xavier_uniform_(matrix, gain = 1)

        emb.weight.data.copy_(matrix)

    def init_multi_gpu(self, device):
        # self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.feature = nn.DataParallel(self.feature, device_ids=device)
        self.moments_lda = nn.DataParallel(self.moments_lda, device_ids=device)


    def forward(self, user):
        article = user['articles']
        moments = user['moments']
        uemb = user['id']
        age = user['age']
        gender = user['gender']

        
        article = self.feature(article)
        moments = self.moments_lda(moments)

        uemb = self.UserEmb(uemb)
        #return self.out(uemb)

        age = self.age(age)
        gender = self.gender(gender)

        # out = torch.cat([uemb, age, gender, article, moments], dim = 1)
        out = torch.cat([uemb, age, gender], dim = 1)
        return self.out(out)


