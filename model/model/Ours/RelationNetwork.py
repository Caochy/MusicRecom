import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result

from .EncoderMusic import User, Item
# from .EncoderAli import User, Item

class AttentionEncoder(nn.Module):
    def __init__(self, config):
        super(AttentionEncoder, self).__init__()
        
        self.emb_size = config.getint('model', 'emb_size')
        self.attn_head = 10

        self.music_encoder = Item(config)
        self.user_encoder = User(config)
                
        self.music_attn = nn.MultiheadAttention(self.emb_size, self.attn_head)
        self.user_attn = nn.MultiheadAttention(self.emb_size, self.attn_head)
        
        # self.interaction = nn.MultiHeadAttention(self.emb_size, self.attn_head)


    def forward(self, music, user, history = False):
        musicv = self.music_encoder(music)
        userv = self.user_encoder(user)

        if history:
            userv = userv.repeat(int(musicv.shape[0]/userv.shape[0]), 1, 1)
        

        tmusic = torch.transpose(musicv, 0, 1)
        tuser = torch.transpose(userv, 0, 1)
        
        uservec, _ = self.user_attn(tuser, tuser, tuser)
        uservec = torch.transpose(uservec, 0, 1)

        uservec, _ = torch.max(uservec, dim = 1)

        # print(uservec.shape)

        musicvec, weight = self.music_attn(torch.transpose(uservec.unsqueeze(1), 0, 1), tmusic, tmusic)
        musicvec = torch.transpose(musicvec, 0, 1)
        musicvec, _ = torch.max(musicvec, dim = 1)

        # print(musicvec.shape)

        
        return torch.cat([musicvec, uservec], dim = 1), weight



class RelationModule(nn.Module):
    def __init__(self, config):
        super(RelationModule, self).__init__()
        
        self.emb_size = config.getint('model', 'emb_size')
        self.memory = nn.Parameter(torch.Tensor(2 * self.emb_size, 2 * self.emb_size))
        
        nn.init.xavier_normal_(self.memory)


    def forward(self, support, query, score):
        ans = support.matmul(self.memory)
        ans = torch.bmm(ans, query.unsqueeze(2)).squeeze(2)
        ans = torch.sigmoid(ans)
        ans = ans # * score
        # ans = torch.softmax(ans, dim = 1)
    
        return torch.bmm(ans.unsqueeze(1), support).squeeze(1), ans


class MultiHeadRelationModule(nn.Module):
    def __init__(self, config):
        super(MultiHeadRelationModule, self).__init__()

        self.emb_size = config.getint('model', 'emb_size')
        self.attn_head = 10

        self.attn = nn.MultiheadAttention(2 * self.emb_size, self.attn_head)

    def forward(self, support, query, score):
        qqq = torch.transpose(query.unsqueeze(1), 0, 1)
        sss = torch.transpose(support, 0, 1)
        ans, weight = self.attn(qqq, sss, sss)
        
        # ans, weight = self.attn(query.unsqueeze(1), support, support)
        return ans.squeeze(0), weight.squeeze(1)


class RelationNetwork(nn.Module):
    def __init__(self, config):
        super(RelationNetwork, self).__init__()
        
        self.emb_size = config.getint('model', 'emb_size')
        
        self.encoder = AttentionEncoder(config)
        self.relation = RelationModule(config)
        # self.relation = MultiHeadRelationModule(config)

        self.out = nn.Linear(self.emb_size * 4, 2)
        # self.out = nn.Linear(self.emb_size * 2, 2)
        self.relu = nn.ReLU(True)

    def init_multi_gpu(self, device):
        return


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        users = data['users']
        candidate = data['candidate']
        history = data['history']
        labels = data['label']
        score = data['score']
        

        candidate, cweight = self.encoder(candidate, users)
        
        batch = labels.shape[0]
        k = history['id'].shape[1]
        for key in history:
            history[key] = history[key].view(batch * k, -1)
        history, hweight = self.encoder(history, users, True)
        history = history.view(batch, k, -1)
        

        interest, similarity = self.relation(history, candidate, score)
        
        # similarity = torch.mean(similarity, dim = 1).unsqueeze(1)
        similarity = torch.max(similarity, dim = 1)[0].unsqueeze(1)
        y1 = torch.cat([1 - similarity, similarity], dim = 1)
        
        y2 = self.out(torch.cat([interest, candidate], dim = 1))
        
        # y2 = self.out(candidate)
        
        loss = criterion(y2, labels) + criterion(y1, labels)#+ self.relu(torch.mean(hweight) - 0.7) # - torch.mean(torch.log(torch.max(hweight.squeeze(), dim = 1)[0]))
        # loss = criterion(y, labels) - torch.mean(torch.log(torch.max(hweight.squeeze(), dim = 1)[0]))
        
        
        y = torch.softmax(y1, dim = 1) + torch.softmax(y2, dim = 1)
        y = y * 0.5
        
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                                "accuracy_result": acc_result}


