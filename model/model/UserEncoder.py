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
        
        # self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.feature = nn.Linear(self.article_feature_len, self.hidden)
        self.moments_lda = nn.Linear(self.moments_lda_len, self.hidden)

    def init_multi_gpu(self, device):
        # self.bert = nn.DataParallel(self.bert, device_ids=device)
        self.feature = nn.DataParallel(self.feature, device_ids=device)
        self.moments_lda = nn.DataParallel(self.moments_lda, device_ids=device)


    def forward(self, user):
        article = user['articles']
        moments = user['moments']

        
        article = self.feature(article)
        moments = self.moments_lda(moments)


        return torch.cat([article, moments], dim = 1)

