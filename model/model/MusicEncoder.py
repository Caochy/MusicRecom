import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from pytorch_pretrained_bert import BertModel, BertForPreTraining
from utils.util import calc_accuracy, gen_result, generate_embedding

class MusicEncoder(nn.Module):
    def __init__(self, config):
        super(MusicEncoder, self).__init__()
        
        self.song_feature_len = config.getint('model', 'song_feature_len')
        self.hidden = config.getint('model', 'hidden_size')
        self.emb_size = config.getint('model', 'emb_size')
        
        #self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.feature = nn.Linear(self.song_feature_len, self.emb_size)
        self.lyric = nn.Linear(768, self.emb_size)

        #self.singers = nn.Linear(417, self.hidden) # th = 20
        self.singers = nn.Embedding(417, self.emb_size)
        #self.genre = nn.Linear(18, self.hidden) # th 100
        self.genre = nn.Embedding(18, self.emb_size)
        
        self.MusicEmb = nn.Embedding(42800, self.emb_size)
        
        feature_size = self.emb_size * 3
        self.out = nn.Linear(feature_size, self.hidden * 2)

        self.init_embedding(self.singers)
        self.init_embedding(self.genre)
        self.init_embedding(self.MusicEmb)


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

        
        # print('singer', singers.shape)
          
        memb = self.MusicEmb(memb).squeeze()

        # return self.out(memb)

        feature = self.feature(feature).squeeze()
        singers = self.singers(singers).squeeze()
        genre = self.genre(genre).squeeze()


        # embs = torch.cat([memb, feature, singers, genre], dim = 1)
        embs = torch.cat([memb, singers, genre], dim = 1)
        
        return self.out(embs)
        
        '''
        _, y = self.bert(lyric, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        
        y = self.lyric(y)
        
        return torch.cat([feature, y], dim = 1)
        '''
        return feature
