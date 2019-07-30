import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding
from model.model.FieldEncoder import FieldEncoder

class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for cross_layer_size in cross_layer_sizes:
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return torch.sum(torch.cat(xs, dim=1), 2)
    
class xDeepFM(nn.Module):
    def __init__(self, config):
        super(xDeepFM, self).__init__()
        self.field_size = config.getint('model', 'hidden_size')
        self.field_num= 9
        self.field_encoder = FieldEncoder(config)
        self.cross_layer_sizes= [32,16,8]
        self.split_half=config.getboolean('model','split_half')
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
        self.CIN=CompressedInteractionNetwork(self.field_num,self.cross_layer_sizes,self.split_half)
        print("deep_layer2:{},final:{}".format(self.deep_layer2,self.field_num+sum(self.cross_layer_sizes)+self.deep_layer2))
        self.final=nn.Linear(self.field_num+sum(self.cross_layer_sizes)+self.deep_layer2,2)
        
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
        # # -----------linear part ------------------------
        self.y_first_order= torch.sum(self.emb,dim=2)
        self.y_first_order= self.dropout(self.y_first_order)
        # # -----------CIN part----------------------------
        y_cin=self.CIN(self.emb)
        # # -----------DNN part----------------------------
        self.y_deep=self.emb.reshape((-1,self.field_num*self.field_size))
        self.y_deep=self.dropout(self.y_deep)
        self.y_deep=self.mlp(self.y_deep)
#         print("self.y_first_order,self.y_cin,self.y_deep:{},{},{}".format(self.y_first_order.shape,self.y_cin.shape,self.y_deep.shape))
        self.output=torch.cat([self.y_first_order,y_cin,self.y_deep],dim=1)
        self.output=self.final(self.output)
        loss = criterion(self.output, label)
        accu, accu_result = calc_accuracy(self.output, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(self.output, dim=1)[1].cpu().numpy(), "x": self.output,
                        "accuracy_result": acc_result}


