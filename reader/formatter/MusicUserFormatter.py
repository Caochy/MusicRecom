import json
import torch
import numpy as np
import os
from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter

class MusicUserFormatter:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = UserFormatter(config)

    
    def check(self, data, config):
        try:
            data = json.loads(data)

            if self.music.check(data) is None or self.user.check(data) is None:
                return None
            return data

        except:
            return None


    def format(self, alldata, config, transformer, mode):
        music = {'lyric': [], 'features': []}
        users = {'articles': [], 'moments': []}
        labels = []
        
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

        return {'music': music, 'users': users, 'label': labels}
  
