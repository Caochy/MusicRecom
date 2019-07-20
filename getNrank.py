import argparse
import os
import torch
from torch import nn

from config_reader.parser import ConfigParser
from model.get_model import get_model
from reader.reader import init_dataset,init_formatter
from model.work import valid_net,DataCuda
from utils.util import print_info
from model.loss import get_loss
import random
import numpy as np
from reader.formatter.MusicPairFormatter import MusicPairFormatter
from reader.formatter.DeepInterestFormatter import DeepInterestFormatter
from reader.formatter.NCFFormatter import NCFFormatter
from reader.formatter.DeepFMFormatter import DeepFMFormatter
class Recommender:
    def __init__(self,config_path,gpu,model_path):
        # gpu
        self.use_gpu = True
        if gpu is None:
            self.use_gpu = False
        else:
            self.use_gpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        
        # config
        self.config = ConfigParser(config_path)
        
        # formatter 
        useable_list = {
                # "AYYC": AYPredictionFormatter
                "DeepFM":DeepFMFormatter,
                "LRMM": MusicPairFormatter,
                "DeepInterest": DeepInterestFormatter,
                "NCF": NCFFormatter
            }
        if self.config.get("data", "formatter") in useable_list.keys():
            self.formatter = useable_list[self.config.get("data", "formatter")](self.config)
        else:
            raise NotImplementedError
        task_loss_type = self.config.get("train", "type_of_loss")
        self.criterion = get_loss(task_loss_type)
        
        #model
        model_name = self.config.get("model", "name")
        net = get_model(model_name, self.config)
        device = []
        if torch.cuda.is_available() and self.use_gpu:
            net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        self.net=net
        print_info("Net build done")
        
    def getNrank(self,uid,ranknum,samplenum):
        uid=int(uid)
        ranknum=int(ranknum)
        samplenum=int(samplenum)
        mcand=random.sample(self.formatter.allmusic,samplenum)
        mcand=[mid for mid in mcand if self.formatter.music.check(mid,self.config)]
        samplenum=len(mcand)
        data=[[[uid for _ in range(samplenum)],mcand,[1 for _ in range(samplenum)]]]

        data=self.formatter.format(data,self.config,None,None)
        with torch.no_grad():
            data = DataCuda(data, self.use_gpu)
            results = self.net(data, self.criterion, self.config, self.use_gpu, [])
            select=torch.argsort(results['x'][:,1]).cpu().numpy()
            mid_nrank=np.asarray(mcand)[select[-ranknum:]]
        return mid_nrank


