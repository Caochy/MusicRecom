import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result



class User(nn.Module):
    def __init__(self, config):
        super(User, self).__init__()

        self.shop_len = 3
        self.cms_group_len = 13
        self.occu_len = 2
        self.city_len = 5
        self.gender_len = 2
        self.pvalue_len = 4
        self.age_len = 7
        self.id_len = 64844

        self.emb_size = config.getint('model', 'emb_size')
        
        self.shop = nn.Embedding(self.shop_len, self.emb_size)
        self.cms = nn.Embedding(self.cms_group_len, self.emb_size)
        self.occu = nn.Embedding(self.occu_len, self.emb_size)
        self.city = nn.Embedding(self.city_len, self.emb_size)
        self.gender = nn.Embedding(self.gender_len, self.emb_size)
        self.pvalue = nn.Embedding(self.pvalue_len, self.emb_size)
        self.age = nn.Embedding(self.age_len, self.emb_size)
        self.id = nn.Embedding(self.id_len, self.emb_size)
        
    
    
    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])
        nn.init.xavier_uniform_(matrix, gain = 1)

        emb.weight.data.copy_(matrix)


    def forward(self, user):
        age = self.age(user['age']).unsqueeze(1)
        pvalue = self.pvalue(user['pvalue']).unsqueeze(1)
        shop = self.shop(user['shop']).unsqueeze(1)
        occu = self.occu(user['occu']).unsqueeze(1)
        city = self.city(user['city']).unsqueeze(1)
        gender = self.gender(user['gender']).unsqueeze(1)
        cms = self.cms(user['cms']).unsqueeze(1)
        ids = self.id(user['id']).unsqueeze(1)

        return torch.cat([ids, age, pvalue, shop, occu, city, gender, cms], dim = 1)


class Item(nn.Module):
    def __init__(self, config):
        super(Item, self).__init__()

        self.cate_len = 806
        self.customer_len = 935
        self.campaign_len = 411
        self.price_len = 11
        self.pids_len = 2
        self.brand_len = 846
        self.id_len = 52343
        
        self.emb_size = config.getint('model', 'emb_size')

        self.cate = nn.Embedding(self.cate_len, self.emb_size) 
        self.customer = nn.Embedding(self.customer_len, self.emb_size)
        self.campaign = nn.Embedding(self.campaign_len, self.emb_size)
        self.price = nn.Embedding(self.price_len, self.emb_size)
        self.pids = nn.Embedding(self.pids_len, self.emb_size)
        self.brand = nn.Embedding(self.brand_len, self.emb_size)
        self.id = nn.Embedding(self.id_len, self.emb_size)


    def init_emb(self, emb):
        matrix = torch.Tensor(emb.weight.shape[0], emb.weight.shape[1])

        nn.init.xavier_uniform_(matrix, gain = 1)
        emb.weight.data.copy_(matrix)


    def forward(self, items):
        cate = self.cate(items['cate']).unsqueeze(1)
        customer = self.customer(items['customer']).unsqueeze(1)
        brand = self.brand(items['brand']).unsqueeze(1)
        campaign = self.campaign(items['campaign']).unsqueeze(1)
        price = self.price(items['price']).unsqueeze(1)
        ids = self.id(items['id']).unsqueeze(1)
        pids = self.pids(items['pids']).unsqueeze(1)

        
        return torch.cat([pids, cate, customer, brand, campaign, price], dim = 1)
        # return torch.cat([ids, pids, cate, customer, brand, campaign, price], dim = 1)

