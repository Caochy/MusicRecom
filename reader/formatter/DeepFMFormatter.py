import json
import torch
import numpy as np
import os
import random
from .MusicFormatter import MusicFormatter
from .UserFormatter import UserFormatter


class DeepFMFormatter:
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
#         testId= data['music']
#         if self.music.check(testId,config) is None:
#             return None
#         testLabel=data['label']
#         musicId= list(data['history'].keys())
#         musicId=[id1 for id1 in musicId if not(self.music.check(id1,config) is None)]
#         musicNum=len(musicId)
#         musicNgNum=int(musicNum*self.ng)
#         if musicNum <1 :
#             return None
        
        
#         musicNGid=[]
#         musicNgCnt=0
#         while musicNgCnt<musicNgNum:
#             id1=random.sample(self.allmusic,1)[0]
#             if self.music.check(id1,config) is None and (id1 not in musicId):
#                 continue    
#             musicNgCnt+=1
#             musicNGid.append(id1)
#         musicId.extend(musicNGid)
#         labels= [1 for _ in range(musicNum)]+[0 for _ in range(int(musicNgNum))]
        
        
#         return [[userid for _ in range(len(labels))],musicId,labels,testId,testLabel]

        musicId= data['music']
        label= data['label']
        if self.music.check(musicId,config) is None:
            return None

        return [userid,musicId,label]

       

    def format(self, alldata, config, transformer, mode):
        
        labels  =[]
        musicids=[]
        userids =[]
#         if mode=='train':
#             for d in alldata:
#                 labels.extend(d[2])
#                 musicids.extend(d[1])
#                 userids.extend(d[0])   
        for d in alldata:
            userids.append(d[0])
            musicids.append(d[1])
            labels.append(d[2])
            
#         elif mode=='valid':
#             for d in alldata:
#                 labels.append(d[4])
#                 musicids.append(d[3])
#                 userids.append(d[0][0])
        music = self.music.format(musicids, config, mode)
        users = self.user.format(userids, config, mode)        
        labels = torch.tensor(labels, dtype = torch.long)
        return {'music': music, 'users': users, 'label': labels}
  
