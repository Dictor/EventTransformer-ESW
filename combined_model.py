import torch
from data_generation import Event_DataModule

import json
from tqdm import tqdm
import time
import numpy as np
import evaluation_utils
from trainer import EvNetModel

# device = 'cuda:0'
device = 'cpu'


# path_model = './pretrained_models/DVS128_10_24ms_dwn/'
# path_model = './pretrained_models/DVS128_11_24ms_dwn/'
# path_model = './pretrained_models/SLAnimals_4s_48ms_dwn/'
# path_model = './pretrained_models/SLAnimals_3s_48ms_dwn/'
path_model = './pretrained_models/ASL_DVS_dwn/'


path_weights = evaluation_utils.get_best_weigths(path_model, 'val_acc', 'max')
#evaluation_utils.plot_training_evolution(path_model)
all_params = json.load(open(path_model + '/all_params.json', 'r'))
model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device('cpu'), **all_params).eval().to(device)

data_params = all_params['data_params']
data_params['batch_size'] = 1
data_params['pin_memory'] = False
data_params['sample_repetitions'] = 1
dm = Event_DataModule(**data_params)
dl = dm.val_dataloader()

y_pred = []

model.train()
torch.set_grad_enabled(True)    
    
for polarity, pixels, labels, acc, brk  in tqdm(dl):
    if polarity is None: continue
    polarity, pixels, labels, acc, brk = polarity.to(device), pixels.to(device), labels.to(device), acc.to(device), brk.to(device)
    embs, clf_logits = model(polarity, pixels)
    
    #training logic
    
    y_pred.append(clf_logits.argmax())
    print("pred: {}, acc: {}, brk: {}".format(y_pred[-1].item(), acc.item(), brk.item()))
