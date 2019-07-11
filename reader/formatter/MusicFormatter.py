import json
import torch
import numpy as np
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer

class MusicFormatter:
    def __init__(self, config):
        song_info_path = config.get('data', 'music_info_path')
        singer_num_path = config.get('data', 'singer_num_path')
        genre_num_path = config.get('data', 'genre_num_path')

        singer_num_th = config.get('train', 'singer_num_th')
        genre_num_th = config.get('train', 'genre_num_th')
        
        '''song information'''
        print('init song information')
        self.song_info = {}
        fin = open(song_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            songid = line['song_id']
            self.song_info[int(songid)] = {'lyric': line['lyric'], 'features': line['features'], 'singer': line['singer_name'], 'genre': line['genre']}
        
        self.lyric_len = config.getint("model", "song_lyric_len")

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.get("model", "bert_path"), 'vocab.txt'))
        

        '''singer2id'''
        singer_num = json.load(open(singer_num_path, 'r'))
        self.singer2id = {}
        for s in singer_num:
            if singer_num[s] < singer_num_th:
                continue
            self.singer2id[s] = len(self.singer2id)


        '''genre2id'''
        genre_num = json.load(open(genre_num_path, 'r'))
        self.genre2id = {}
        for g in genre_num:
            if genre_num[g] < genre_num_th:
                continue
            self.genre2id[g] = len(self.genre2id)


    def lookup(self, text, max_len):
        token = self.tokenizer.tokenize(text)
        token = ["[CLS]"] + token
        while len(token) < max_len:
            token.append("[PAD]")
        token = token[0:max_len]
        token = self.tokenizer.convert_tokens_to_ids(token)
        return token


    def check(self, songid, config):
        try:
            #data = json.loads(data)
            if songid not in self.song_info:
                return None
            if self.song_info[songid]['features'] is None:
                return None
            if len(self.song_info[songid]['features']) == 0:
                return None
            return True
        except:
            return None


    def format(self, songids, config, mode):
        music = {'lyric': [], 'features': [], 'singer': [], 'genre': []}

        
        for data in songids:
            music['features'].append(self.song_info[data]['features'])
            music['lyric'].append(self.lookup(self.song_info[data]['lyric'], self.lyric_len))
            
            singer = np.zeros(len(self.singer2id))
            singer[self.singer2id[data['singer_name']]] = 1
            music['singer'].append(singer)

            genre = np.zeros(len(self.genre2id))
            genre[self.genre2id[data['genre']]] = 1
            music['genre'].append(genre)
        
        
        music['features'] = torch.tensor(music['features'], dtype = torch.float32)
        music['lyric'] = torch.tensor(music['lyric'], dtype = torch.long)
        music['singer'] = torch.tensor(music['singer'], dtype = torch.long)
        music['genre'] = torch.tensor(music['genre'], dtype = torch.long)
        

        return music
  
