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

        self.score_th = 8
        self.allmusic = list(self.music.song_info.keys())
        random.shuffle(self.allmusic)
        self.index = 0
        self.fout = None        

    def check(self, data, config):
        
        data = json.loads(data)
        userid = int(data['user'])
        if self.user.check(userid, config) is None:
            return None

        like_music = []

        other_music = []
        for m in data['music']:
            if self.music.check(m, config) is None:
                continue
            if data['music'][m] >= 3 and data['music'][m] < 15:
                like_music.append((m, data['music'][m]))
            if data['music'][m] >= 1:
                other_music.append(m)

        if len(like_music) < self.k + 1:
            return None

        if len(other_music) < 1:
            return None
            
        # random.shuffle(like_music)
        # random.shuffle(dislike_music)

        history = [d[0] for d in like_music[:self.k]]
        score = [d[1] for d in like_music[:self.k]]
        score = [d/24 + 0.5 for d in score]
        if random.randint(0, 1000) % 2 == 0:
            candidate = like_music[-1][0]
            label = 1
        else:
            # candidate = dislike_music[-1]
            while self.music.check(self.allmusic[self.index], config) is None:
                self.index += 1
            candidate = self.allmusic[self.index]
            self.index = (self.index + 1) % len(self.allmusic)
            label = 0


        '''
        candidate = dislike_music
        if len(like_music) > self.k:
            candidate += like_music[self.k:]

        random.shuffle(candidate)
        candidate = candidate[0]
        ''' 
        
        return {'user': userid, 'history': history, 'candidate': candidate, 'label': label, 'score': score}



    def format(self, alldata, config, transformer, mode):
        '''        
        if mode == 'train':
            if not self.fout is None:
                self.fout.close()
                self.fout = None
        else:
            if self.fout is None:
                self.fout = open('test.txt', 'w')
            for d in alldata:
                print('%s\t%d' % (d['candidate'], d['label']), file = self.fout)
        '''
        
        users = self.user.format([u['user'] for u in alldata], config, mode)
        candidate = self.music.format([u['candidate'] for u in alldata], config, mode)

        history = self.music.format_history([u['history'] for u in alldata], config, mode)
        score = torch.tensor([d['score'] for d in alldata], dtype = torch.float)
        
        '''
        print('history data infomation')
        for key in history:
            print(key, history[key].shape)
        '''
        
        labels = torch.tensor([d['label'] for d in alldata], dtype = torch.long)


        return {'candidate': candidate, 'users': users, 'label': labels, 'history': history, 'score': score}
  
