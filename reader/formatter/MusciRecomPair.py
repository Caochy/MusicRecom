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
            self.user_info[line['id']] = {'article': line['article']}

        print('init song information')
        self.song_info = {}
        fin = open(song_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            songid = line['song_id']
            self.song_info[songid] = {'lyric': line['lyric'], 'features': line['features']}
        

        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.song_len = config.getint("data", "song_len")


    def lookup(self, data, max_len):
        lookup_id = []
        for word in data:
            try:
                lookup_id.append(self.word2id[word])
            except:
                # bert
                # lookup_id.append(self.word2id["[UNK]"])
                lookup_id.append(self.word2id["UNK"])

        while len(lookup_id) < max_len:
            lookup_id.append(self.word2id["PAD"])
        lookup_id = lookup_id[:max_len]
        
        return lookup_id
           

    def check(self, data, config):
        try:
            data = json.loads(data)
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

        for data in alldata:
            music['features'].append(self.song_info[data[1]]['features'])
            users['articles'].append(self.user_info[data[0]]['article'])

            labels.append(data[2])

        music['features'] = torch.tensor(music['features'], dtype = torch.float32)
        music['lyric'] = torch.tensor(music['lyric'], dtype = torch.long)

        users['articles'] = torch.tensor(users['articles'], dtype = torch.float32)
        users['moments'] = torch.tensor(users['moments'], dtype = torch.long)

        labels = torch.tensor(labels, dtype = torch.long)

        return {'music': music, 'users': users, 'label': labels}



            
