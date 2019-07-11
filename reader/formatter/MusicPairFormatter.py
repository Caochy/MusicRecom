import json
import torch
import numpy as np
import os

from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter


class MusicPairFormatter:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = MusicFormatter(config)
        

    def check(self, data, config):
        try:
            data = json.loads(data)

            data[0] = int(data[0])
            data[1] = int(data[1])
            if self.user.check(data[0]):
                return None
            if self.music.check(data[1]):
                return None
            return data
        except:
            return None


    def format(self, alldata, config, transformer, mode):
        music = {'lyric': [], 'features': []}
        users = {'articles': [], 'moments': []}
        labels = []

        musicids = [d[1] for d in alldata]
        userids = [d[0] for d in alldata]

        labels = []
        for d in alldata:
            if d[2] > 4.5:
                labels.append(1)
            else:
                labels.append(0)
        music = self.music.format(musicids, config, mode)
        users = self.users.format(userids, config, mode)

        
        '''
        label2id = {'dislike': 0, 'like': 1}
        for data in alldata:
            music['features'].append(self.song_info[data[1]]['features'])
            music['lyric'].append(self.lookup(self.song_info[data[1]]['lyric'], self.lyric_len))

            users['moments'].append(self.user_info[data[0]]['moments_lda'])
            users['articles'].append(self.user_info[data[0]]['article'])
            
            labels.append(label2id[data[2]])

        music['features'] = torch.tensor(music['features'], dtype = torch.float32)
        music['lyric'] = torch.tensor(music['lyric'], dtype = torch.long)

        users['articles'] = torch.tensor(users['articles'], dtype = torch.float32)
        users['moments'] = torch.tensor(users['moments'], dtype = torch.float32)

        labels = torch.tensor(labels, dtype = torch.long)
        '''

        return {'music': music, 'users': users, 'label': labels}
  
