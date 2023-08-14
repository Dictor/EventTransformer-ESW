import scipy.io
import os
from tqdm import tqdm
import sparse
import numpy as np
import pickle
import csv
import math


# Source data folder
path_dataset = '../datasets/ESW/'
dataset_name = ['back6']
# Target data folder
path_dataset_dst = '../datasets/ESW/clean_dataset_frames/'


chunk_len_ms = 2
chunk_len_us = chunk_len_ms*1000
source_width = 640; source_height = 320
target_width = 240; target_height = 180 
width_divider = source_width / target_width; height_divider = source_height / target_height; 

for fn in dataset_name:
    fn_dst = path_dataset_dst + '/{}.pckl'.format(fn)
    if os.path.isfile(fn_dst): print("target already exist")
    
    reader = csv.reader(open(os.path.join(path_dataset, '{}.txt'.format(fn)), "rt"), delimiter=' ', quoting=csv.QUOTE_NONE)
    mat = {'ts': [], 'x': [], 'y': [], 'pol': []}
    
    for row, record in enumerate(reader):
        if len(record) == 2: continue
        mat['ts'].append(int(float(record[0]) * 1000))
        mat['x'].append(math.floor(int(record[1]) / width_divider))
        mat['y'].append(math.floor(int(record[2]) / height_divider))
        mat['pol'].append(int(record[3]))
        #if row > 5000 : break
                
    if mat['ts'][-1] < mat['ts'][0] > 200*1000: print("suspicious ts", mat['ts'][-1])
    
    intg = np.array([mat['x'], mat['y'], mat['ts'], mat['pol']])
    print(intg.shape)
    total_events = intg.transpose()
    print(total_events.shape)
    
    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4: 
            pass
        else:
            total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:chunk_inds.min()]
    print('chunk count : ', len(total_chunks))
    total_chunks = total_chunks[::-1]
    print("")
        
    total_frames = []
    for chunk in total_chunks:
        debug = chunk[:,[1,0,3]]
        frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'), 
                           np.ones(chunk.shape[0]).astype('int32'), 
                           (target_height, target_width, 2))   # .to_dense()
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)
    
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')    
    
    if len(total_frames) > 200*1000 / chunk_len_us: print("many frames", mat['ts'][-1])
    
    pickle.dump(total_frames, open(fn_dst, 'wb'))