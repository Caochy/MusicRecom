import json
import torch
import numpy as np


class MusicPairFormatter:
    def __init__(self, config):
        user_info_path = config.get('data', 'user_info_path')
        song_info_path = config.get('data', 'music_info_path')
        
        print('init user infomation')
        self.user_info = {}
        fin = open(user_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            self.user_info[int(line['id'])] = {'article': line['article'], 'moments_lda': line['moments_lda']}

        print('init song information')
        self.song_info = {}
        fin = open(song_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            songid = line['song_id']
            self.song_info[int(songid)] = {'lyric': line['lyric'], 'features': line['features']}
        
        
        self.word2id = {}
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        c = 0
        for line in f:
            self.word2id[line[:-1]] = c
            c += 1
        # self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.lyric_len = config.getint("data", "song_lyric_len")


    def lookup(self, data, max_len):
        lookup_id = []
        for word in data:
            try:
                lookup_id.append(self.word2id[word])
            except:
                # bert
                lookup_id.append(self.word2id["[UNK]"])
                # lookup_id.append(self.word2id["UNK"])

        while len(lookup_id) < max_len:
            lookup_id.append(self.word2id["[PAD]"])
        lookup_id = lookup_id[:max_len]
        
        return lookup_id
           

    def check(self, data, config):
        try:
            data = json.loads(data)

            data[0] = int(data[0])
            data[1] = int(data[1])
            if data[0] not in self.user_info:
                return None
            if data[1] not in self.song_info:
                return None
            return data
        except:
            return None


    def format(self, alldata, config, transformer, mode):
        music = {'lyric': [], 'features': []}
        users = {'articles': [], 'moments': []}
        labels = []
        
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

        return {'music': music, 'users': users, 'label': labels}
  
