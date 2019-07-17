import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding
from model.model.FieldEncoder import FieldEncoder
class DeepFM(nn.Module):
    def __init__(self, config):
        super(DeepFM, self).__init__()
        self.field_size = config.getint('model', 'hidden_size')
        self.field_num= 9
        self.field_encoder = FieldEncoder(config)
        self.deep_layer1=config.getint('model','deep_layer1')
        self.deep_layer2=config.getint('model','deep_layer2')
        self.dropout=nn.Dropout()
        self.mlp=nn.Sequential(
                    nn.Linear(self.field_num*self.field_size,self.deep_layer1),
                    nn.BatchNorm1d(self.deep_layer1),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(self.deep_layer1,self.deep_layer2),
                    nn.BatchNorm1d(self.deep_layer2),
                    nn.ReLU(),
                    nn.Dropout()
                )
        self.final=nn.Linear(self.field_size+self.field_num+self.deep_layer2,2)
        
    def init_multi_gpu(self, device):
#         self.field_encoder = nn.DataParallel(self.field_encoder,device)
#         self.mlp=nn.DataParallel(self.mlp,device)
        pass

    def forward(self, data, criterion, config, usegpu, acc_result = None):
        user = data['users']
        music = data['music']
        label = data['label']
        self.emb = self.field_encoder(user,music) # None*(F*K)
        self.emb = self.emb.reshape((-1,self.field_num,self.field_size)) # None *F *K
        # # -----------FM part------------------------------
        # ------------first order term---------
        self.y_first_order= torch.sum(self.emb,dim=2)
        self.y_first_order= self.dropout(self.y_first_order)
        # ---------- second order term ---------------
        summed=torch.sum(self.emb,dim=1)
        self.summed_square=torch.mul(summed,summed) #None *K
        self.squared_sum=torch.sum(torch.mul(self.emb,self.emb),dim=1) #None *K
        self.y_second_order=0.5*(self.summed_square-self.squared_sum) # None*K
        self.y_second_order=self.dropout(self.y_second_order)
        # # -----------DNN part----------------------------
        self.y_deep=self.emb.reshape((-1,self.field_num*self.field_size))
        self.y_deep=self.dropout(self.y_deep)
        self.y_deep=self.mlp(self.y_deep)
        self.output=torch.cat([self.y_first_order,self.y_second_order,self.y_deep],dim=1)
        self.output=self.final(self.output)
        loss = criterion(self.output, label)
        accu, accu_result = calc_accuracy(self.output, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(self.output, dim=1)[1].cpu().numpy(), "x": self.output,
                        "accuracy_result": acc_result}


