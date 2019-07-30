import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding

class FieldEncoder(nn.Module):
    def __init__(self, config):
        super(FieldEncoder, self).__init__()
        # user
        self.article_feature_len = config.getint('model', 'user_article_len')
        self.moments_lda_len = config.getint('model', 'moments_lda_len')
        self.hidden = config.getint('model', 'hidden_size')
        self.ufeature = nn.Linear(self.article_feature_len, self.hidden)
        self.moments_lda = nn.Linear(self.moments_lda_len, self.hidden)
        self.UserEmb = nn.Embedding(190662, self.hidden)
        self.age = nn.Embedding(6, self.hidden)
        self.gender = nn.Embedding(2, self.hidden)
        
        # music
        self.song_feature_len = config.getint('model', 'song_feature_len')
        self.sfeature = nn.Linear(self.song_feature_len, self.hidden)
        self.singers = nn.Embedding(417, self.hidden)
        self.genre = nn.Embedding(18, self.hidden)
        self.MusicEmb = nn.Embedding(42800, self.hidden)
        
        self.init_emb(self.UserEmb)
        self.init_emb(self.age)
        self.init_emb(self.gender)
        self.init_emb(self.singers)
        self.init_emb(self.genre)
        self.init_emb(self.MusicEmb)
        
        self.moments_batchnorm = nn.BatchNorm1d(self.moments_lda_len)
        self.article_batchnorm = nn.BatchNorm1d(self.article_feature_len)
        self.batchnorm = nn.BatchNorm1d(self.song_feature_len)
        
    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])
        nn.init.xavier_uniform_(matrix, gain = 1)
        emb.weight.data.copy_(matrix)
        
    def init_multi_gpu(self, device):
        self.ufeature = nn.DataParallel(self.ufeature, device_ids=device)
        self.moments_lda = nn.DataParallel(self.moments_lda, device_ids=device)
        self.sfeature=nn.DataParallel(self.sfeature,device=device)

    def forward(self, user,music):
        # user
        uemb = user['id']
        uemb = self.UserEmb(uemb)
        memb = music['id']
        memb = self.MusicEmb(memb)
#         return torch.cat([uemb,memb], dim = 1)
        
        article = user['articles']
        age = user['age']
        gender = user['gender']
        article =self.article_batchnorm(article)
        article = self.ufeature(article)
        
        age = self.age(age)
        gender = self.gender(gender)
        
        
        # music
        singers = music['singer']
        genre = music['genre']
        
        singers = self.singers(singers)
        genre = self.genre(genre)
        
        
#         return torch.cat([uemb, age, gender, article,singers,genre,memb], dim = 1)
        
        
        moments = user['moments']
        feature = music['features']
        feature= self.batchnorm(feature)
        feature = self.sfeature(feature)
        moments = self.moments_batchnorm(moments)
        moments = self.moments_lda(moments)
        out = torch.cat([uemb, age, gender, article, moments,feature,singers,genre,memb], dim = 1)
        return out


