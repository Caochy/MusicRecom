import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding

from model.model.MusicEncoder import MusicEncoder
from model.model.UserEncoder import UserEncoder


class LatentRelMM(nn.Module):
    def __init__(self, config):
        super(LatentRelMM, self).__init__()
        
        self.N = config.getint('model', 'memory_N')
        
        self.hidden = config.getint('model', 'hidden_size') * 2

        self.user_encoder = UserEncoder(config)
        self.music_encoder = MusicEncoder(config)

        self.memory = nn.Parameter(torch.Tensor(self.N, self.hidden))
        self.out = nn.Linear(self.hidden, 2)
        self.th = 5


    def init_multi_gpu(self, device):
        self.user_encoder = nn.DataParallel(self.user_encoder)
        self.music_encoder = nn.DataParallel(self.music_encoder)
        # self.memory = nn.DataParallel(self.memory)
        self.out = nn.DataParallel(self.out)

    def forward(self, data, criterion, config, usegpu, acc_result = None):
        user = data['users']
        music = data['music']

        label = data['label']

        user = self.user_encoder(user)
        music = self.music_encoder(music)

        s = user * music
        
        out_result = self.out(s)

        s = s.matmul(torch.transpose(self.memory, 0, 1))
        s = torch.softmax(s, dim = 1)
        rel = s.matmul(self.memory)
        

        score = user + rel - music
        
        #print(score.shape)

        score = torch.norm(score, dim = 1)
        mask = (2 * label - 1).float()
        
        #print(mask.shape)
        #print(score.shape)

        loss = torch.mean(mask * score) # + criterion(out_result, label)

        accu, accu_result = calc_accuracy(out_result, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result,
                        "accuracy_result": acc_result}


