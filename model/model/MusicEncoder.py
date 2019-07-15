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
        
        #self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.feature = nn.Linear(self.song_feature_len, self.hidden)
        self.lyric = nn.Linear(768, self.hidden)

        #self.singers = nn.Linear(417, self.hidden) # th = 20
        self.singers = nn.Embedding(417, self.hidden)
        #self.genre = nn.Linear(18, self.hidden) # th 100
        self.genre = nn.Embedding(18, self.hidden)
        
        self.MusicEmb = nn.Embedding(42800, self.hidden)
        

        self.out = nn.Linear(4 * self.hidden, self.hidden * 2)

    
    def forward_history(self, history):
        batch = history['id'].shape[0]
        k = history['id'].shape[1]

        for key in history:
            # print(key, batch, k, history[key].shape)
            
            history[key] = history[key].view(batch * k, -1)

        out = self.forward(history)
        out = out.view(batch, k, self.hidden * 2)
        
        return out


    def forward(self, music):
        lyric = music['lyric']
        feature = music['features']
        
        singers = music['singer']
        genre = music['genre']

        memb = music['id']

        
        #print('singer', singers.shape)

        feature = self.feature(feature).squeeze()
        singers = self.singers(singers).squeeze()
        genre = self.genre(genre).squeeze()

        memb = self.MusicEmb(memb).squeeze()

        embs = torch.cat([memb, feature, singers, genre], dim = 1)
        return self.out(embs)
        
        '''
        _, y = self.bert(lyric, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        
        y = self.lyric(y)
        
        return torch.cat([feature, y], dim = 1)
        '''
        return feature
