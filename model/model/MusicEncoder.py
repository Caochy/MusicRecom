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

        self.singers = nn.Linear(417, self.hidden) # th = 20
        self.genre = nn.Linear(18, self.hidden) # th 100

        self.out = nn.Linear(3 * self.hidden, self.hidden * 2)

    def init_multi_gpu(self, device):
        #self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.feature = nn.DataParallel(self.feature, device_ids=device)
        self.lyric = nn.DataParallel(self.lyric, device_ids=device)


    def forward(self, music):
        lyric = music['lyric']
        feature = music['features']
        
        singers = music['singer']
        genre = music['genre']



        feature = self.feature(feature)
        singers = self.singers(singers)
        genre = self.genre(genre)
        
        embs = torch.cat([feature, singers, genre], dim = 1)
        return self.out(embs)
        
        '''
        _, y = self.bert(lyric, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        
        y = self.lyric(y)
        
        return torch.cat([feature, y], dim = 1)
        '''
        return feature
