import json
import torch
import numpy as np
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

class MusicFormatter:
    def __init__(self, config):
        item_info_path = config.get('data', 'item_info_path')
        with open(item_info_path,"r") as file:
            line=json.loads(file.readline())
            self.item_keys=list(line.keys())
        self.num_feat=len(self.item_keys)
        self.featname2idv=np.vectorize(dict(zip(self.item_keys,range(self.num_feat))).get)
        print('init song information')
        item_info=[]
        prices =[]
        with open(item_info_path,"r") as file:
            for line in file:
                line=json.loads(line)
                values=list(line.values())
                item_cats=list(map(int,values[:-1]))
                prices.append(float(values[-1]))
                item_info.append(item_cats)
        self.item_info=np.asarray(item_info)
        self.prices=np.asarray(prices)
        item_set=[list(set(self.item_info[:,i])) for i in range(self.num_feat-1)]  #[userid(),...()]
        self.item_map=[dict(zip(item_set[i],range(len(item_set[i])))) for i in range(self.num_feat-1)] #[{"$id1":0,"$id2":1...}...]
        self.item2id=self.item_map[self.item_keys.index('adgroup_id')]
        self.item2idv=np.vectorize(self.item2id.get)
        for i in range(self.num_feat-1):
            self.item_info[:,i]=np.vectorize(self.item_map[i].get)(self.item_info[:,i])
        self.num_item=len(self.item2id)
        print("item num : {}".format(self.num_item))
        
    def check(self, data, config):
        try:
            if data not in self.item2id:
                return None
            return True
        except Exception as err:
            print(err)
            return None
    def format(self, songids, config, mode):
        item_select=self.item2idv(songids)
#         feat_select=self.featname2idv(["adgroup_id","brand"])
        cat=[torch.tensor(self.item_info[item_select,i],dtype=torch.long) for i in range(self.num_feat-1)]
        price=torch.tensor(self.prices[item_select],dtype=torch.float32).unsqueeze(1)
        return [cat,price]
  
