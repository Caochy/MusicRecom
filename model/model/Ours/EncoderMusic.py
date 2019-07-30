import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result



class User(nn.Module):
    def __init__(self, config):
        super(User, self).__init__()
        
        self.article_feature_len = config.getint('model', 'user_article_len')
        self.moments_lda_len = config.getint('model', 'moments_lda_len')
        self.hidden = config.getint('model', 'hidden_size')
        self.emb_size = config.getint('model', 'emb_size')
        
        self.feature = nn.Linear(self.article_feature_len, self.emb_size)
        self.moments_lda = nn.Linear(self.moments_lda_len, self.emb_size)

        self.UserEmb = nn.Embedding(190662, self.emb_size)
        self.age = nn.Embedding(6, self.emb_size)
        self.gender = nn.Embedding(2, self.emb_size)
        

        self.moments_batchnorm = nn.BatchNorm1d(self.moments_lda_len)
        self.article_batchnorm = nn.BatchNorm1d(self.article_feature_len)


    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])
        nn.init.xavier_uniform_(matrix, gain = 1)

        emb.weight.data.copy_(matrix)


    def forward(self, user):
        article = user['articles']
        moments = user['moments']
        uemb = user['id']
        age = user['age']
        gender = user['gender']
        
        
        article = self.article_batchnorm(article)
        moments = self.moments_batchnorm(moments)
        
        article = self.feature(article).unsqueeze(1)
        moments = self.moments_lda(moments).unsqueeze(1)
        

        uemb = self.UserEmb(uemb).unsqueeze(1)
        age = self.age(age).unsqueeze(1)
        gender = self.gender(gender).unsqueeze(1)
        
        # return torch.cat([uemb, age, gender], dim = 1)
        # return torch.cat([age, gender, article, moments], dim = 1)
        return torch.cat([uemb, age, gender, article, moments], dim = 1)


class Item(nn.Module):
    def __init__(self, config):
        super(Item, self).__init__()
        self.song_feature_len = config.getint('model', 'song_feature_len')
        self.hidden = config.getint('model', 'hidden_size')
        self.emb_size = config.getint('model', 'emb_size')
        
        self.feature = nn.Linear(self.song_feature_len, self.emb_size)
        self.lyric = nn.Linear(768, self.emb_size)

        self.singers = nn.Embedding(417, self.emb_size)
        self.genre = nn.Embedding(18, self.emb_size)

        self.MusicEmb = nn.Embedding(42800, self.emb_size)
        

        self.init_embedding(self.singers)
        self.init_embedding(self.genre)
        self.init_embedding(self.MusicEmb)
        
        self.batchnorm = nn.BatchNorm1d(self.song_feature_len)


    def init_embedding(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])

        nn.init.xavier_uniform_(matrix, gain = 1)
        emb.weight.data.copy_(matrix)


    def forward(self, music):
        lyric = music['lyric']
        feature = music['features']
        singers = music['singer']
        genre = music['genre']
        memb = music['id']
        
        feature = self.batchnorm(feature)

        memb = self.MusicEmb(memb).squeeze().unsqueeze(1)
        feature = self.feature(feature).squeeze().unsqueeze(1)
        singers = self.singers(singers).squeeze().unsqueeze(1)
        genre = self.genre(genre).squeeze().unsqueeze(1)
        
        # return torch.cat([memb, singers, genre], dim = 1)
        return torch.cat([feature, singers, genre], dim = 1)
        # return torch.cat([memb, feature, singers, genre], dim = 1)

