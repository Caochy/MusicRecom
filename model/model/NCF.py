import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json
import os
from utils.util import calc_accuracy, gen_result, generate_embedding
class NCF(nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.model = config.get('model','modelStage')
        print("ModelStage:",self.model)
        if self.model == 'NeuMF-pre': 
            self.GMF_model_path = config.get('model','GMF_model')
            if os.path.exists(self.GMF_model_path):
                config.config.set('model','modelStage','GMF')
                self.GMF_model=NCF(config).cuda()
                self.GMF_model.load_state_dict(torch.load(self.GMF_model_path))
                print("load GMF done")
            else:print("no GMF_model:",self.GMF_model)
            
            self.MLP_model_path = config.get('model','MLP_model')
            if os.path.exists(self.MLP_model_path):
                config.config.set('model','modelStage','MLP')
                self.MLP_model=NCF(config).cuda()
                self.MLP_model.load_state_dict(torch.load(self.MLP_model_path))
                print("load MLP done")
            else:print("no MLP_model:",self.MLP_model)
        

        user_num=190662
        item_num=42800
        factor_num=config.getint('model','factor_num')
        num_layers=config.getint('model','num_layers')
        self.dropout=config.getfloat('model','dropout')
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 2)

        self._init_weight_()

    def _init_weight_(self):
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                            self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                            self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                            self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                            self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                        self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result = None):
        user=data['users']
        item=data['music']
        label=data['label']
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)
        self.output = self.predict_layer(concat) 

        loss = criterion(self.output, label)
        accu, accu_result = calc_accuracy(self.output, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(self.output, dim=1)[1].cpu().numpy(), "x": self.output,
                        "accuracy_result": acc_result}


