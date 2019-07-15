import json
import torch
import numpy as np
import os
from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter
import random


class DeepInterestFormatter:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = UserFormatter(config)
        
        self.k = config.getint('train', 'num_interaction')
        
    def check(self, data, config):
        
        data = json.loads(data)
        userid = int(data['user'])
        if self.user.check(userid, config) is None:
            return None

        like_music = []
        dislike_music = []
        for m in data['music']:
            if self.music.check(m, config) is None:
                continue
            if data['music'][m] > 4.5:
                like_music.append(m)
            else:
                dislike_music.append(m)

        if len(like_music) < self.k + 5:
            return None

        if len(dislike_music) < 5:
            return None
            
        random.shuffle(like_music)
        random.shuffle(dislike_music)

        history = like_music[:self.k]
        
        if random.randint(0, 1000) % 2 == 0:
            candidate = like_music[-1]
        else:
            candidate = dislike_music[0]
        '''
        candidate = dislike_music
        if len(like_music) > self.k:
            candidate += like_music[self.k:]

        random.shuffle(candidate)
        candidate = candidate[0]
        ''' 
        if data['music'][candidate] > 4.5:
            label = 1
        else:
            label = 0

        return {'user': userid, 'history': history, 'candidate': candidate, 'label': label}



    def format(self, alldata, config, transformer, mode):
        
        users = self.user.format([u['user'] for u in alldata], config, mode)
        candidate = self.music.format([u['candidate'] for u in alldata], config, mode)

        history = self.music.format_history([u['history'] for u in alldata], config, mode)

        labels = torch.tensor([d['label'] for d in alldata], dtype = torch.long)


        return {'candidate': candidate, 'users': users, 'label': labels, 'history': history}
  