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
            self.user_info[int(line['id'])] = {'article': line['article'], 'moments_lda': line['moments_lda'], 'age': int(line['age']), 'gender': int(line['gender'])}
        
        self.user2id = {}
        '''user2id'''
        for u in self.user_info:
            self.user2id[u] = len(self.user2id)

        print('the number of users: ', len(self.user2id))


    def check(self, data, config):
        try:
            # data = json.loads(data)

            data = int(data)
            if data not in self.user_info:

                return None
            return True
        except Exception as err:
            print(err)
            return None

    def age2id(self, uid):
        age = self.user_info[int(uid)]['age']
        if age <= 10:
            return 0
        elif age <= 20:
            return 1
        elif age <= 30:
            return 2
        elif age <= 40:
            return 3
        elif age <= 50:
            return 4
        else:
            return 5


    def format(self, userids, config, mode):
        users = {'articles': [], 'moments': [], 'id': [], 'age': [], 'gender': []}
        
        for data in userids:
            data = int(data)
            users['moments'].append(self.user_info[data]['moments_lda'])
            users['articles'].append(self.user_info[data]['article'])
            users['id'].append(self.user2id[data])
            users['age'].append(self.age2id(data))
            
            users['gender'].append(self.user_info[data]['gender'] - 1)


        
        users['articles'] = torch.tensor(users['articles'], dtype = torch.float32)
        users['moments'] = torch.tensor(users['moments'], dtype = torch.float32)
        users['id'] = torch.tensor(users['id'], dtype = torch.long)
        users['age'] = torch.tensor(users['age'], dtype = torch.long)
        users['gender'] = torch.tensor(users['gender'], dtype = torch.long)

        return users
  
