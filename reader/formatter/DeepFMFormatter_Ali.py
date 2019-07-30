import json
import torch
import numpy as np
import os
import random
from .MusicFormatter_Ali import MusicFormatter
from .UserFormatter_Ali import UserFormatter


class DeepFMFormatter_Ali:
    def __init__(self, config):
        self.music = MusicFormatter(config)
        self.user = UserFormatter(config)
    
    def check(self, data, config):
        
        data = json.loads(data)
        userid = int(data['userid'])
        if self.user.check(userid, config) is None:
            return None
        testId=int(data['candidate'])
        if self.music.check(testId,config) is None:
            return None
        testLabel=int(data['label'])
        
        return [userid,testId,testLabel]

       

    def format(self, alldata, config, transformer, mode):
        alldata=np.asarray(alldata)
        userids,musicids,labels  = list(map(np.squeeze,np.hsplit(alldata,3)))
        music = self.music.format(musicids, config, mode)
        users = self.user.format(userids, config, mode)        
        labels = torch.tensor(labels, dtype = torch.long)
        return {'music': music, 'users': users, 'label': labels}
  
