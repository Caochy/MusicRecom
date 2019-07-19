#from model.model.LSTM import LSTM
#from model.model.TextCNN import TextCNN
#from model.model.basic_model import CV_pay

from model.model.LRMM import LatentRelMM
from model.model.DeepInterest import DeepInterest
from model.model.DeepFM import DeepFM
from model.model.NCF import NCF
from model.model.Ours.RelationNetwork import RelationNetwork

model_list = {
    #"LSTM": LSTM,
    #"TextCNN": TextCNN,
    "LRMM": LatentRelMM,
    "DeepInterest": DeepInterest,
    "DeepFM":DeepFM,
    "NCF":NCF,
    "RelationNetwork": RelationNetwork
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
