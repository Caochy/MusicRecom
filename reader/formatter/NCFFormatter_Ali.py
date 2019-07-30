import json
import torch
import numpy as np
import os
import random
from .MusicFormatter_Ali import MusicFormatter
from .UserFormatter_Ali import UserFormatter


class NCFFormatter_Ali:
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
        userids,itemids,labels  = np.hsplit(alldata,3)
        music=torch.tensor(self.music.item2idv(itemids),dtype=torch.long).squeeze()
        users=torch.tensor(self.user.user2idv(userids),dtype=torch.long).squeeze()
        labels = torch.tensor(labels, dtype = torch.long).squeeze()
        return {'music': music, 'users': users, 'label': labels}
  
