import json
import torch
import numpy as np
import os

from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter


class NCFFormatter:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = UserFormatter(config)
        

    def check(self, data, config):
        try:
            data = json.loads(data)
            data[0] = int(data[0])
            data[1] = int(data[1])
            if self.user.check(data[0], config) is None:
                return None
            if self.music.check(data[1], config) is None:
                return None
            return data
        except Exception as err:
            print(err)
            return None


    def format(self, alldata, config, transformer, mode):
        labels = []

        musicids = [d[1] for d in alldata]
        userids = [d[0] for d in alldata]

        labels = []
        for d in alldata:
            if d[2] > 4.5:
                labels.append(1)
            else:
                labels.append(0)
        music = []
        users = []
        for id1 in musicids:
            music.append(self.music.song2id[id1])
        for id1 in userids:
            users.append(self.user.user2id[id1])
        labels = torch.tensor(labels, dtype = torch.long)
        music=torch.tensor(music,dtype=torch.long)
        users=torch.tensor(users,dtype=torch.long)
        return {'music': music, 'users': users, 'label': labels}
  
