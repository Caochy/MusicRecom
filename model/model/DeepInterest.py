import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result

from model.model.MusicEncoder import MusicEncoder
from model.model.UserEncoder import UserEncoder


class ActivationUnit(nn.Module):
    def __init__(self, config):
        super(ActivationUnit, self).__init__()

        self.hidden = config.getint('model', 'hidden_size')
        self.linear = nn.Linear(self.hidden * 4, 1)
        self.relu = nn.ReLU()


    def forward(self, history, candidate):
        cand = candidate.unsqueeze(1).repeat(1, history.shape[1], 1)

        #out = cand * history
        out = torch.cat([cand, history], dim = 2)
        
        # print(out.shape)

        out = self.linear(out) # batch, k, 1
        
        weight = torch.sigmoid(out)
        
                
        out = torch.bmm(torch.transpose(weight, 1, 2), history)
        out = out.squeeze(1)
        # out = self.prelu(out)

        return out#, weight.squeeze(1)


class DeepInterest(nn.Module):
    def __init__(self, config):
        super(DeepInterest, self).__init__()

        self.hidden = config.getint('model', 'hidden_size')
        
        self.user_encoder = UserEncoder(config)
        self.music_encoder = MusicEncoder(config)
        self.act = ActivationUnit(config)
        
        '''
        self.out = nn.Sequential(
            nn.Linear(self.hidden * 6, self.hidden),
            nn.ReLU(True),
            nn.Linear(self.hidden, 2)
        )
        '''
        #self.out1 = nn.Linear(self.hidden * 6, self.hidden)
        self.out1 = nn.Linear(self.hidden * 4, 2)
        self.out2 = nn.Linear(self.hidden, 2)

        self.relu = nn.ReLU(True)

        nn.init.xavier_normal_(self.out1.weight)
        nn.init.xavier_normal_(self.out2.weight)


    def init_multi_gpu(self, device):
        return
        self.user_encoder = nn.DataParallel(self.user_encoder, device)
        self.music_encoder = nn.DataParallel(self.music_encoder, device)
        self.act  = nn.DataParallel(self.act, device)
        self.out = nn.DataParallel(self.out)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        users = data['users']
        candidate = data['candidate']
        history = data['history']
        labels = data['label']
        

        users = self.user_encoder(users)
        #if acc_result is None:
        #    print(users)
        candidate = self.music_encoder(candidate)

        
        batch = labels.shape[0]
        k = history['id'].shape[1]
        for key in history:
            history[key] = history[key].view(batch * k, -1)
        history = self.music_encoder(history)
        history = history.view(batch, k, -1)
        
        
        sumpooling = self.act(history, candidate)
        
        # feature = torch.cat([users, sumpooling, candidate], dim = 1)
        feature = torch.cat([users, candidate], dim = 1)
        #print(feature.shape)

        #y = self.out2(self.relu(self.out1(feature)))
        y = self.out1(feature)
        
        # y = torch.softmax(y, dim = 1)
        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                        "accuracy_result": acc_result}





