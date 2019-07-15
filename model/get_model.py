#from model.model.LSTM import LSTM
#from model.model.TextCNN import TextCNN
#from model.model.basic_model import CV_pay

from model.model.LRMM import LatentRelMM
from model.model.DeepInterest import DeepInterest


model_list = {
    #"LSTM": LSTM,
    #"TextCNN": TextCNN,
    "LRMM": LatentRelMM,
    "DeepInterest": DeepInterest
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
