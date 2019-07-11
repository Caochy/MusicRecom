import json
import torch
import numpy as np
import os

class UserFormatter:
    def __init__(self, config):
        user_info_path = config.get('data', 'user_info_path')
        
        print('init user infomation')
        self.user_info = {}
        fin = open(user_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            self.user_info[int(line['id'])] = {'article': line['article'], 'moments_lda': line['moments_lda']}


    def check(self, data, config):
        try:
            # data = json.loads(data)

            data[0] = int(data[0])
            if data[0] not in self.user_info:
                return None
        except:
            return None


    def format(self, userids, config, mode):
        users = {'articles': [], 'moments': []}
        
        for data in userids:
            users['moments'].append(self.user_info[data]['moments_lda'])
            users['articles'].append(self.user_info[data]['article'])
        
        
        users['articles'] = torch.tensor(users['articles'], dtype = torch.float32)
        users['moments'] = torch.tensor(users['moments'], dtype = torch.float32)
        

        return users
  
