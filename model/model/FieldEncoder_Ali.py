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
        self.hidden = config.getint('model', 'hidden_size')        
        self.user_cat=json.loads(config.get('model','user_cat'))
        self.user_cat_emb=nn.ModuleList([nn.Embedding(i,self.hidden) for i in self.user_cat])
        
        for i in range(len(self.user_cat)):
            self.init_emb(self.user_cat_emb[i])
            
        self.item_cat=json.loads(config.get('model','item_cat'))
        self.item_cat_emb=nn.ModuleList([nn.Embedding(i,self.hidden) for i in self.item_cat])
        for i in range(len(self.item_cat)):
            self.init_emb(self.item_cat_emb[i])
        
        self.item_con_emb=nn.Linear(1,self.hidden)
        
    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])
        nn.init.xavier_uniform_(matrix, gain = 1)
        emb.weight.data.copy_(matrix)
        
    def init_multi_gpu(self, device):
        pass

    def forward(self, user,item):
        
        out_user_cat=[self.user_cat_emb[i](user[i]) for i in range(len(self.user_cat))]
        item_cat,item_con=item
        out_item_cat=[self.item_cat_emb[i](item_cat[i]) for i in range(len(self.item_cat))]
        #out_item_con=self.item_con_emb(item_con)
        
        out=[]
        out.extend(out_user_cat)
        out.extend(out_item_cat)
        #out.append(out_item_con)
        out = torch.cat(out, dim = 1)
        return out


