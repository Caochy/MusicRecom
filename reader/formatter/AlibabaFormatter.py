import json
import torch
import numpy as np
import os
import random


class AdLoader:
    def __init__(self, config):
        ad_info_path = config.get('data', 'ad_info_path')
        self.ad_info = {}
        fin = open(ad_info_path, 'r')
        for line in fin:
            line = json.loads(line)
            self.ad_info['adgroup_id'] = line
        
        fin = open(config.get('data', 'ad_info2id_path'), 'r')
        self.customer2id = json.loads(fin.readline())
        self.brand2id = json.loads(fin.readline())
        self.campaign2id = json.loads(fin.readline())
        self.cate2id = json.loads(fin.readline())
        self.adid2id = json.loads(fin.readline())
        self.pid2id = {
            '430548_1007': 0,
            '430539_1007': 1
        } 
    def price2id(self, price):
        if price < 10:
            return 0
        elif price < 30:
            return 1
        elif price < 60:
            return 2
        elif price < 100:
            return 3
        elif price < 150:
            return 4
        elif price < 200:
            return 5
        elif price < 250:
            return 6
        elif price < 300:
            return 7
        elif price < 400:
            return 8
        elif price < 600:
            return 9
        else:
            return 10
    

    def change2id(self, dic, key):
        if key in dic:
            return dic[key]
        else:
            return len(dic)


    def format(self, adids, pids):
        ans = {'cate': [], 'customer': [], 'brand': [], 'campaign': [], 'price': [], 'id': [], 'pids': [self.pid2id[v] for v in pids]}
        for ad in adids:
            ans['cate'].append(self.change2id(self.cate2id, self.ad_info[ad]['cate_id']))
            ans['customer'].append(self.change2id(self.customer2id, self.ad_info[ad]['customer']))
            ans['brand'].append(self.change2id(self.brand2id, self.ad_info[ad]['brand']))
            ans['campaign'].append(self.change2id(self.campaign2id, self.ad_info[ad]['campaign_id']))
            ans['price'].append(self.price2id(self.ad_info[ad]['price']))
            ans['id'].appens(self.adid2id[ad])

        for key in ans:
            ans[key] = torch.tensor(ans[key], dtype = torch.long)

        return ans


class UserLoader:
    def __init__(self, config):
        user_info_path = config.get('data', 'user_info_path')
        fin = open(user_info_path, 'r')
        self.user_info = {}

        for line in fin:
            line = json.loads(line)
            self.user_info[line['userid']] = line

        fin = open(config.get('data', 'user_info2id_path'), 'r')
        self.userid2id = json.loads(fin.readline())

    def format(self, userids):
        ans = {'age': [], 'pvalue': [], 'shop': [], 'occu': [], 'city': [], 'gender': [], 'cms': [], 'id': []}
        for u in userids:
            ans['age'].append(int(self.user_info[u]['age_level']))
            ans['pvalue'].append(int(self.user_info[u]['pvalue_level']) - 1)
            ans['shop'].append(int(self.user_info[u]['shopping_level']) - 1)
            ans['occu'].append(int(self.user_info[u]['occupation']))
            city = int(self.user_info[u]['new_user_class_level'])
            if city < 0:
                city += 1
            ans['city'].append(city)
            ans['gender'].append(int(self.user_info[u]['final_gender_code']) - 1)
            ans['cms'].append(int(self.user_info[u]['cms_group_id']))
            ans['id'].append(self.userid2id[u])

        for key in ans:
            ans[key] = torch.tensor(ans[key], dtype = torch.long)
        return ans



class AlibabaFormatter:
    def __init__(self, config):
        
        self.user = UserLoader(config)
        self.ad = AdLoader(config)

    def check(self, data, config):
        data = json.loads(data)
        return data


    def format(self, alldata, config, transformer, mode):
        users = self.user.format([d['userid'] for d in alldata])
        candidate = self.ad.format([d['candidate'] for d in all_data], [d['candidate_pid'] for d in all_data])

        history = [d['history'] for d in alldata]
        history_res = [self.ad.format([d['candidate'] for d in his], [d['pid'] for d in his]) for his in history]
        
        history_ans = {}
        keys = ['cate', 'customer', 'brand', 'campaign', 'price', 'id', 'pids']
        for key in keys:
            history_ans[key] = torch.cat([d[key].unsqueeze(0) for d in history_res], dim = 0)



        return {'candidate': candidate, 'users': users, 'label': labels, 'history': history_ans, 'score': score}
  
