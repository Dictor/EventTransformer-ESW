import torch
from data_generation import Event_DataModule

from pytorch_lightning.metrics import Accuracy
import pandas as pd
import json
from tqdm import tqdm
import time
import numpy as np

import evaluation_utils
from trainer import EvNetModel

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import os
import pickle
import json
from skimage.util import view_as_blocks
# import copy
# from scipy import ndimage

from neural_compressor.experimental import Quantization
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

# device = 'cuda'
device = 'cpu'
# torch.set_num_threads(2)
path_model = './pretrained_models/ESW/'


path_weights = evaluation_utils.get_best_weigths(path_model, 'val_acc', 'max')
all_params = json.load(open(path_model + 'all_params.json', 'r'))
model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device(device), **all_params).eval().to(device)

def get_params(model):
    total_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().sum().iloc[0]
    pos_encoding_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().loc['pos_encoding'].iloc[0]
    stats = {
        'total_params': total_params,
        'backbone_params': total_params - pos_encoding_params,
        'pos_encoding_params': pos_encoding_params
        }
    return stats

print('\n\n ** Calculating parameter statistics')
param_stats = get_params(model)

data_params = all_params['data_params']
data_params['batch_size'] = 1
data_params['pin_memory'] = False
data_params['sample_repetitions'] = 1
dm = Event_DataModule(**data_params)
dl = dm.val_dataloader()




# conf = (
#     PostTrainingQuantConfig(
#         # device = "gpu",
#         tuning_criterion=TuningCriterion(
#             timeout=0,
#             max_trials=100,
#         ),
#         approach="dynamic")
# )  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")
# q_model = quantization.fit(
#     model=model,
#     conf=conf,
#     calib_dataloader=dl,
# )
# q_model.save('./quantizer/output')

class Top1Metric(object):
    def __init__(self):
        self.correct = 0
    def update(self, output, label):
        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(label.view_as(pred)).sum().item()
    def reset(self):
        self.correct = 0
    def result(self):
        return 100. * self.correct / len(dl.dataset)

quantizer = Quantization("./conf.yaml")
quantizer.model = model
quantizer.calib_dataloader = dl
quantizer.eval_dataloader = dl
# quantizer.metric = Top1Metric()
q_model = quantizer()
q_model.save('./quantizer/output')