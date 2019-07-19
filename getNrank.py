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
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--gpu', '-g')
parser.add_argument('--model', '-m')
parser.add_argument('--userid','-u')
parser.add_argument('--num','-n')
args = parser.parse_args()

configFilePath = args.config
if configFilePath is None:
    print("python *.py\t--config/-c\tconfigfile")
use_gpu = True

if args.gpu is None:
    use_gpu = False
else:
    use_gpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = ConfigParser(configFilePath)

print_info("Start to build Net")

model_name = config.get("model", "name")
net = get_model(model_name, config)

device = []
if torch.cuda.is_available() and use_gpu:
    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        device.append(int(a))

    net = net.cuda()

    try:
        net.init_multi_gpu(device)
    except Exception as e:
        print_info(str(e))

net.load_state_dict(torch.load(args.model))

print_info("Net build done")
with open("../predata/DataUser/train.json") as file:
    data=file.readline()
from reader.formatter.MusicPairFormatter import MusicPairFormatter
from reader.formatter.DeepInterestFormatter import DeepInterestFormatter
from reader.formatter.NCFFormatter import NCFFormatter
from reader.formatter.DeepFMFormatter import DeepFMFormatter
useable_list = {
        # "AYYC": AYPredictionFormatter
        "DeepFM":DeepFMFormatter,
        "LRMM": MusicPairFormatter,
        "DeepInterest": DeepInterestFormatter,
        "NCF": NCFFormatter
    }
if config.get("data", "formatter") in useable_list.keys():
    formatter = useable_list[config.get("data", "formatter")](config)
else:
    raise NotImplementedError

task_loss_type = config.get("train", "type_of_loss")
criterion = get_loss(task_loss_type)

uid=int(args.userid)
ranknum=int(args.num)
samplenum=ranknum*100


mcand=random.sample(formatter.allmusic,samplenum)
mcand=[mid for mid in mcand if formatter.music.check(mid,config)]
samplenum=len(mcand)
print("samplenum:",samplenum)
data=[[[uid for _ in range(samplenum)],mcand,[1 for _ in range(samplenum)]]]

data=formatter.format(data,config,None,None)
with torch.no_grad():
        data = DataCuda(data, use_gpu)
        results = net(data, criterion, config, use_gpu, [])
        select=torch.argsort(results['x'][:,1]).cpu().numpy()
        print(np.asarray(mcand)[select[-ranknum:]])


