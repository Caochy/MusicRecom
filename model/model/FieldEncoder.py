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
    def init_multi_gpu(self, device):
        self.ufeature = nn.DataParallel(self.ufeature, device_ids=device)
        self.moments_lda = nn.DataParallel(self.moments_lda, device_ids=device)
        self.sfeature=nn.DataParallel(self.sfeature,device=device)

    def forward(self, user,music):
        # user
        article = user['articles']
        moments = user['moments']
        uemb = user['id']
        age = user['age']
        gender = user['gender']
        article = self.ufeature(article)
        moments = self.moments_lda(moments)
        uemb = self.UserEmb(uemb)
        age = self.age(age)
        gender = self.gender(gender)
        # music
        feature = music['features']
        singers = music['singer']
        genre = music['genre']
        memb = music['id']

        feature = self.sfeature(feature)
        feature=feature.squeeze()
        
        singers = self.singers(singers).squeeze()
        genre = self.genre(genre).squeeze()
        memb = self.MusicEmb(memb).squeeze()

        
        out = torch.cat([uemb, age, gender, article, moments,feature,singers,genre,memb], dim = 1)
        return out


