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
import copy
from scipy import ndimage

import time

# import sys
# import select
# import tty
# import termios
# from pynput.keyboard import Key, Listener
# from pynput import keyboard
# from curtsies import Input

#device = 'cuda'
device = 'cpu'
torch.set_num_threads(2)
path_model = './pretrained_models/ESW/'


path_weights = evaluation_utils.get_best_weigths(path_model, 'val_acc', 'max')
all_params = json.load(open(path_model + '/all_params.json', 'r'))
model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device(device), **all_params).eval().to(device)


def shift(all_params, total_pixels, cropped_shape):
    height = all_params["backbone_params"]["pos_encoding"]["params"]["shape"][1]
    width = all_params["backbone_params"]["pos_encoding"]["params"]["shape"][0]
    validation = all_params["backbone_params"]["pos_enc_grad"]
    patch_size = all_params["data_params"]["patch_size"]

    height_diff, width_diff = height - cropped_shape[0], width - cropped_shape[1]
    if not validation:
        new_height_init = np.random.randint(0, height_diff) if height_diff != 0.0 else 0
        new_width_init = np.random.randint(0, width_diff)  if width_diff != 0.0 else 0
    else:
        new_height_init, new_width_init = height_diff // 2, width_diff // 2
        
    new_height_init -= new_height_init % patch_size    #; new_height_init += self.patch_size//2
    new_width_init -= new_width_init % patch_size      #; new_width_init += self.patch_size//2
    
    for i in range(len(total_pixels)): 
        total_pixels[i][:, 0] += new_height_init
        total_pixels[i][:, 1] += new_width_init
    return total_pixels
def crop_in_time(all_params, total_events):
    validation = all_params["backbone_params"]["pos_enc_grad"]
    augmentation_params = all_params["data_params"]["augmentation_params"]
    num_sparse_frames = augmentation_params['max_sample_len_ms']
    # print('Cropping:', len(total_events))
    if len(total_events) > num_sparse_frames:
        if not validation:     # Crop sequence randomly
            init = np.random.randint(len(total_events) - num_sparse_frames)
            end = init + num_sparse_frames
            total_events = total_events[init:end]
        else:                       # Crop to the middle part
            init = (len(total_events) - num_sparse_frames) // 2
            end = init + num_sparse_frames
            total_events = total_events[init:end]
    # assert len(total_events) < num_sparse_frames, str(len(total_events)) + '  ' + str(num_sparse_frames)
    return total_events
def crop_in_space(all_params, total_events):
    validation = all_params["backbone_params"]["pos_enc_grad"]
    patch_size = all_params["data_params"]["patch_size"]
    height = all_params["backbone_params"]["pos_encoding"]["params"]["shape"][1]
    width = all_params["backbone_params"]["pos_encoding"]["params"]["shape"][0]
    augmentation_params = all_params["data_params"]["augmentation_params"]
    x_lims = (int(width*augmentation_params['random_frame_size']), width)
    y_lims = (int(height*augmentation_params['random_frame_size']), height)

    _, y_size, x_size, _ = total_events.shape
    if not validation:     # Crop sequence randomly
        new_x_size = np.random.randint(x_lims[0], x_lims[1]+1)
        new_y_size = np.random.randint(y_lims[0], y_lims[1]+1)
        
        if patch_size != 1:
            new_x_size -= new_x_size % patch_size
            new_y_size -= new_y_size % patch_size
        
        x_init = np.random.randint(x_size - new_x_size+1); x_end = x_init + new_x_size
        y_init = np.random.randint(y_size - new_y_size+1); y_end = y_init + new_y_size
        total_events = total_events[:, y_init:y_end, x_init:x_end, :]
    else:                       # Crop to the middle part
        new_x_size = (x_lims[0] + x_lims[1])//2
        new_y_size = (y_lims[0] + y_lims[1])//2
        
        if patch_size != 1:
            new_x_size -= new_x_size % patch_size
            new_y_size -= new_y_size % patch_size
            
        x_init = (x_size - new_x_size)//2; x_end = x_init + new_x_size
        y_init = (y_size - new_y_size)//2; y_end = y_init + new_y_size
        total_events = total_events[:, y_init:y_end, x_init:x_end, :]
    assert total_events.shape[1] == new_y_size and total_events.shape[2] == new_x_size, print(total_events.shape, new_y_size, new_x_size)
    return total_events

def getitem(all_params, total_events, total_pdls, return_sparse_array=False,):
    chunk_len_ms = all_params["data_params"]["chunk_len_ms"]
    chunk_len_us = chunk_len_ms*1000
    sparse_frame_len_us = 1000     # len of each loaded sparse frame
    sparse_frame_len_ms = sparse_frame_len_us / 1000                  # self.sparse_frame_len_us//1000 -> self.sparse_frame_len_us/1000
    assert  chunk_len_us % sparse_frame_len_us == 0
    chunk_size = chunk_len_us // sparse_frame_len_us             # Size of the grouped frame chunks
    preproc_polarity = all_params["data_params"]["preproc_polarity"]
    num_extra_chunks = all_params["data_params"]["num_extra_chunks"]
    bins = all_params["data_params"]["bins"]
    patch_size = all_params["data_params"]["patch_size"]
    original_event_size = 1 if '1' in preproc_polarity else 2
    preproc_event_size = original_event_size*bins
    if all_params["data_params"]["min_activations_per_patch"] > 0 and all_params["data_params"]["min_activations_per_patch"] <= 1: 
            min_activations_per_patch = int(all_params["data_params"]["min_activations_per_patch"]*patch_size*patch_size+1)
    else: min_activations_per_patch = 0
    min_patches_per_chunk = all_params["data_params"]["min_patches_per_chunk"]
    augmentation_params = all_params["data_params"]["augmentation_params"]
    validation = all_params["backbone_params"]["pos_enc_grad"]
    h_flip = augmentation_params.get('h_flip', False)


    acc = total_pdls[0]
    brk = total_pdls[1]

    # Crop sequence to self.num_sparse_frames
    if 'max_sample_len_ms' in augmentation_params and augmentation_params['max_sample_len_ms'] != -1:
        total_events = crop_in_time(all_params, total_events)
    if 'random_frame_size' in augmentation_params and augmentation_params['random_frame_size'] is not None:
        total_events = crop_in_space(all_params, total_events)
    if not validation and h_flip and np.random.rand() > 0.5: total_events = total_events[:,:,::-1,:]

    total_pixels, total_polarity = [], []
    if sum(acc)/len(acc)<0.05 and sum(brk)/len(brk)<0.05:
        label = 1 
    elif sum(acc)/len(acc) > sum(brk)/len(brk):
        label = 0
    elif sum(acc)/len(acc) < sum(brk)/len(brk):
        label = 2   
    else:
        label = 1
        #print("ambiguous pedal data, cannot assign label: avg acc={}, avg brk={}".format(sum(acc)/len(acc), sum(brk)/len(brk)))
    current_chunk = None
    # Iterate until read all the total_events (max_sample_len_ms)
    sf_num = len(total_events) - 1
    while sf_num >= 0:
        # Get chunks by grouping sparse frames
        if current_chunk is None: 
            current_chunk = total_events[max(0, sf_num-chunk_size):sf_num][::-1]
            current_chunk = current_chunk.todense()
            sf_num -= chunk_size
            if '1' in preproc_polarity: current_chunk = current_chunk.sum(-1, keepdims=True)
        else:
            sf = total_events[max(0, sf_num-num_extra_chunks):sf_num][::-1]
            sf = sf.todense()
            sf_num -= num_extra_chunks
            if '1' in preproc_polarity: sf = sf.sum(-1, keepdims=True)
            current_chunk = np.concatenate([current_chunk, sf])
        if current_chunk.shape[0] < bins: continue
        # if current_chunk.shape[0] >= self.bins:
        # Divide time-window into bins
        bins_init = current_chunk.shape[0];
        bins_step = bins_init//bins
        chunk_candidate = []
        for ib_num, i in enumerate(list(range(0, bins_init, bins_step))[:bins]):
            if ib_num == bins-1: step = 99999
            else: step = bins_step
            chunk_candidate.append(current_chunk[i:i+step].sum(0))
        chunk_candidate = np.stack(chunk_candidate, axis=-1).astype(float)
        chunk_candidate = chunk_candidate.reshape(chunk_candidate.shape[0], chunk_candidate.shape[1], chunk_candidate.shape[2]*chunk_candidate.shape[3])
        # chunk_candidate = np.array(chunk_candidate)
        # print (chunk_candidate)
        block_shape = np.array((patch_size,patch_size, preproc_event_size))
        arr_shape = np.array(chunk_candidate.shape)
        if (arr_shape % block_shape).sum() != 0:  
            print(block_shape,arr_shape)  
        # Extract patches
        polarity = view_as_blocks(chunk_candidate, block_shape=(patch_size,patch_size, preproc_event_size)); 
        # aggregate by pixel (unique), by patch (sum) -> get the ones with >= min_activations | (num_patches, bool)
        inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], patch_size*patch_size) \
            .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= min_activations_per_patch
        
        if inds.sum() == 0: continue
        # Check if chunk has the desired patch activations and #events
        if min_patches_per_chunk and inds.sum() < min_patches_per_chunk: continue
    
        # Reshape to (num_patches x token_dim)
        polarity = polarity.reshape(polarity.shape[0] * polarity.shape[1], patch_size*patch_size*preproc_event_size)   # self.token_dim
        # Get pixel locations
        pixels = np.array([ (i+patch_size//2,j+patch_size//2) for i in range(0, chunk_candidate.shape[0], patch_size) for j in range(0, chunk_candidate.shape[1], patch_size) ])
        
        inds = np.where(inds)[0]
        
        # Drop patch tokens
        # Apply over the final patch-tokens
        if not validation and len(inds)>0 and 'drop_token' in augmentation_params and augmentation_params['drop_token'][0] != 0.0:
            inds = np.random.choice(inds, replace=False, size=max(1, int(len(inds)*(1-augmentation_params['drop_token'][0]))))
        polarity, pixels = polarity[inds], pixels[inds]

        if 'log' in preproc_polarity: polarity = np.log(polarity + 1)
        else: raise ValueError('Not implemented', preproc_polarity)
            
        assert len(pixels) > 0 and len(polarity) > 0
        total_polarity.append(torch.tensor(polarity))
        total_pixels.append(torch.tensor(pixels).long())   # append(torch.tensor(pixels).long())
        current_chunk = None


    if 'random_shift' in augmentation_params and augmentation_params['random_shift']:
        total_pixels = shift(all_params, total_pixels, total_events.shape[1:-1])
        
    return total_polarity, total_pixels, label      #, acc, brk

def pad_list_of_sequences(samples, token_size, pre_padding = True):
    max_timesteps = max([ len(s) for s in samples ])
    batch_size = len(samples)
    max_event_num = max([ chunk.shape[0] for sample in samples for chunk in sample ])
    
    batch_data = torch.zeros(max_timesteps, batch_size, max_event_num, token_size)
    for num_sample, action_sample in enumerate(samples):
        num_chunks = len(action_sample)
        for chunk_num, chunk in enumerate(action_sample):
            chunk_events = chunk.shape[0]
            if chunk_events == 0:
                continue
            if pre_padding: batch_data[-(num_chunks-chunk_num), num_sample, -chunk_events:, :] = chunk
            else: batch_data[chunk_num, num_sample, :chunk_events, :] = chunk
            
    return batch_data

def collate(batch_samples):
    pols, pixels, labels = [], [], []
    for num_sample, sample in enumerate(batch_samples): 
        if sample is None or len(sample[0]) == 0: 
            continue
        pols.append(sample[0])
        pixels.append(sample[1])
        labels.append(sample[2]) 
    if len(pols) == 0: return None, None, None
    token_size = pols[0][0].shape[-1]
        
    pols = pad_list_of_sequences(pols, token_size, True)
    pixels = pad_list_of_sequences(pixels, 2, True)


    pols, pixels, labels = pols, pixels.long(), torch.tensor(labels).long()
    # acc, brk = torch.tensor(acc).double(), torch.tensor(brk).double()
    return pols, pixels, labels                     #, acc, brk

# label = self.labels[idx]
# def is_pressed():
#     return select.select([sys.stdin],[],[],0)==([sys.stdin],[],[])


# k=0
# def on_press(key):
#     print('{0} pressed'.format(
#         key))
#     k=key
# def on_release(key):
#     print('{0} release'.format(
#         key))
#     return False
    # if key == Key.esc:
    #     # Stop listener
    #     return False

# Load sparse matrix
test_events = pickle.load(open("./datasets/ESW/clean_dataset_1000/train/back6_0.pckl", 'rb'))  # events (t x H x W x 2)
test_pdls = pickle.load(open("./datasets/ESW/clean_dataset_1000/pdl/back6_pedal_0.pckl", 'rb'))  # events (t x H x W x 2)
pre_polarity, pre_pixels, pre_label = getitem(all_params,test_events,test_pdls)
test_samples = [(pre_polarity, pre_pixels, pre_label)]
test_pols, test_pixels, test_labels = collate(test_samples)
# if test_pols is None: continue
test_pols, test_pixels, test_labels = test_pols.to(device), test_pixels.to(device), test_labels.to(device)
t=time.time()
test_clf_logits = model(test_pols, test_pixels)   #test_embs, test_clf_logits = 
print("normal_model_time : ", time.time()-t)

print("test_labels : ", test_labels[0])
print("clf_logits : ", test_clf_logits.argmax())

samples_folder = "/media/soshymking/F41F-86F0/train/"
samples = os.listdir(samples_folder)
inference_time=[]
for i in range(0,1000):
    # path = "./datasets/ESW/clean_dataset_1000_old/train/back1_"
    total_events = pickle.load(open(samples_folder+samples[i], 'rb'))
    filenamesplit = samples[i].split("_")
    total_pdls = pickle.load(open(samples_folder+"../pdl/"+filenamesplit[0]+"_pedal_"+filenamesplit[1], 'rb'))

    pre_pol, pre_pix, pre_lb = getitem(all_params,total_events,total_pdls)
    batch_samples = [(pre_pol, pre_pix, pre_lb)]
    pols, pixels, labels = collate(batch_samples)
    # if test_pols is None: continue
    pols, pixels, labels = pols.to(device),pixels.to(device), labels.to(device)
    t=time.time()
    test_clf_logits = model(pols, pixels)   #test_embs, test_clf_logits = 
    inference_time.append(time.time()-t)

    print("normal_model_time : ", inference_time[i])
    # print(f"pred : {test_clf_logits.argmax()}") #, end='\r'ans : {test_labels[0]}  in : {pdl_in}
    # print("ans : ", test_labels[0], "pred : ", test_clf_logits[1].argmax(),"in : ", pdl_in)
df = pd.DataFrame(inference_time)
df.to_csv("inference_times.csv",index=False)


# Collect events until released
