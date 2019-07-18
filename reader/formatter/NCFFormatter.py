import json
import torch
import numpy as np
import os
import random
from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter


class NCFFormatter:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = UserFormatter(config)
        self.allmusic = list(self.music.song_info.keys())
        self.ng=config.getfloat('model',"num_ng")

    def check(self, data, config):
        data = json.loads(data)
        userid = int(data['user'])
        if self.user.check(userid, config) is None:
            return None

        musicId= list(data['music'].keys())
        musicId=[id1 for id1 in musicId if not(self.music.check(id1,config) is None)]
        musicNum=len(musicId)
        musicNgNum=int(musicNum*self.ng)
        if musicNum <1 :
            return None
        musicNGid=[]
        musicNgCnt=0
        while musicNgCnt<musicNgNum:
            id1=random.sample(self.allmusic,1)[0]
            if self.music.check(id1,config) is None:
                continue    
            musicNgCnt+=1
            musicNGid.append(id1)
        musicId.extend(musicNGid)
        labels= [1 for _ in range(musicNum)]+[0 for _ in range(int(musicNgNum))]
        return [[userid for _ in range(len(labels))],musicId,labels]

    def format(self, alldata, config, transformer, mode):
        labels  =[]
        musicids=[]
        userids =[]
        for d in alldata:
            labels.extend(d[2])
            musicids.extend(d[1])
            userids.extend(d[0])
        music = []
        users = []
        for id1 in musicids:
            music.append(self.music.song2id[int(id1)])
        for id1 in userids:
            users.append(self.user.user2id[id1])
        labels = torch.tensor(labels, dtype = torch.long)
        music=torch.tensor(music,dtype=torch.long)
        users=torch.tensor(users,dtype=torch.long)
        return {'music': music, 'users': users, 'label': labels}
  
