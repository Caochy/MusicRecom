import json
import torch
import numpy as np
import os

class UserFormatter:
    def __init__(self, config):
        user_info_path = config.get('data', 'user_info_path')
        with open(user_info_path,"r") as file:
            line=json.loads(file.readline())
            self.user_keys=list(line.keys())
        self.num_feat=len(self.user_keys)
        self.featname2idv=np.vectorize(dict(zip(self.user_keys,range(self.num_feat))).get)

        print('init user infomation')
        user_info=[]
        with open(user_info_path,"r") as file:
            for line in file:
                line=json.loads(line)
                values=list(map(int,line.values()))
                user_info.append(values)
        self.user_info=np.asarray(user_info)
        user_set=[list(set(self.user_info[:,i])) for i in range(self.num_feat)]  #[userid(),...()]
        self.user_map=[dict(zip(user_set[i],range(len(user_set[i])))) for i in range(self.num_feat)] #[{"$id1":0,"$id2":1...}...]
        self.user2id=self.user_map[0]
        self.user2idv=np.vectorize(self.user_map[0].get)
        for i in range(self.num_feat):
            self.user_info[:,i]=np.vectorize(self.user_map[i].get)(self.user_info[:,i])
        self.num_user=len(self.user2id)
        print('the number of users: ', self.num_user)
    def check(self, data, config):
        try:
            # data = json.loads(data)
            data = int(data)
            if data not in self.user2id:
                return None
            return True
        except Exception as err:
            print(err)
            return None
    def format(self, userids, config, mode):
        user_select=self.user2idv(userids)
#         feat_select=self.featname2idv(["userid","shopping_level"])
        return torch.tensor(self.user_info[user_select,:],dtype=torch.long)
        
  
