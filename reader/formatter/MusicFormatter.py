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

        singer_num_th = config.getint('train', 'singer_num_th')
        genre_num_th = config.getint('train', 'genre_num_th')
        
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
        
        print('the number of song: ', len(self.song_info))
        
        '''song2id'''
        self.song2id = {}
        for key in self.song_info:
            self.song2id[key] = len(self.song2id)


        '''singer2id'''
        singer_num = json.load(open(singer_num_path, 'r'))
        self.singer2id = {}
        for s in singer_num:
            if singer_num[s] < singer_num_th:
                continue
            self.singer2id[s] = len(self.singer2id)

        print('singer2id: ', len(self.singer2id))


        '''genre2id'''
        genre_num = json.load(open(genre_num_path, 'r'))
        self.genre2id = {}
        for g in genre_num:
            if genre_num[g] < genre_num_th:
                continue
            self.genre2id[g] = len(self.genre2id)
        print('genre2id: ', len(self.genre2id))



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
            songid = int(songid)
            #data = json.loads(data)
            if songid not in self.song_info:
                return None
            if self.song_info[songid]['features'] is None:
                return None
            if len(self.song_info[songid]['features']) == 0:
                return None
            return True
        except Exception as err:
            print(err)
            return None


    def format_history(self, history_songids, config, mode):
        musics = [self.format(ids, config, mode) for ids in history_songids]
        
        ans = {}
        keys = ['lyric', 'features', 'singer', 'genre', 'id']
        
        for key in keys:
            ans[key] = torch.cat([d[key].unsqueeze(0) for d in musics], dim = 0)
            # print(ans[key].shape)
        
        return ans


    def format(self, songids, config, mode):
        music = {'lyric': [], 'features': [], 'singer': [], 'genre': [], 'id': []}

        
        for data in songids:
            data = int(data)
            music['id'].append(self.song2id[data])

            data_info = self.song_info[data]
            music['features'].append(data_info['features'])
            music['lyric'].append(self.lookup(data_info['lyric'], self.lyric_len))
            
            if data_info['singer'] in self.singer2id:
                music['singer'].append(self.singer2id[data_info['singer']])
            else:
                music['singer'].append(len(self.singer2id))

            if data_info['genre'] in self.genre2id:
                music['genre'].append(self.genre2id[data_info['genre']])
            else:
                music['genre'].append(len(self.genre2id))

            '''
            singer = np.zeros(len(self.singer2id) + 1)
            if data_info['singer'] in self.singer2id:
                singer[self.singer2id[data_info['singer']]] = 1
            else:
                singer[len(self.singer2id)] = 1
            music['singer'].append(singer)

            genre = np.zeros(len(self.genre2id) + 1)
            if data_info['genre'] in self.genre2id:
                genre[self.genre2id[data_info['genre']]] = 1
            else:
                genre[len(self.genre2id)] = 1
            music['genre'].append(genre)
            '''
        
        
        music['features'] = torch.tensor(music['features'], dtype = torch.float32)
        music['lyric'] = torch.tensor(music['lyric'], dtype = torch.long)
        #music['singer'] = torch.tensor(music['singer'], dtype = torch.float32)
        #music['genre'] = torch.tensor(music['genre'], dtype = torch.float32)
        
        music['singer'] = torch.tensor(music['singer'], dtype = torch.long)
        music['genre'] = torch.tensor(music['genre'], dtype = torch.long)
        
        music['id'] = torch.tensor(music['id'], dtype = torch.long)

        return music
  
