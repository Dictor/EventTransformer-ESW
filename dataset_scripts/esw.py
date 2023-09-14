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
path_dataset_dst = '../datasets/clean_dataset_frames/'


chunk_len_us = 500 # 1000fps
file_len_event = 5000000 
source_width = 640; source_height = 480
target_width = 240; target_height = 180 
width_divider = source_width / target_width; height_divider = source_height / target_height; 

for fn in dataset_name:
    path = os.path.join(path_dataset, '{}.txt'.format(fn))
    print("start to open data : {}".format(fn))
    reader = csv.reader(open(path, "rt"), delimiter=' ', quoting=csv.QUOTE_NONE)
    mat = {'ts': [], 'x': [], 'y': [], 'pol': []}

    with open(path) as csv_file:
        lines = len(csv_file.readlines())
    
    print("complete to make file handle")
    for row, record in enumerate(tqdm(reader, total=lines)):
        if len(record) == 2: continue
        mat['ts'].append(int(float(record[0]) * 1000))
        mat['x'].append(math.floor(int(record[1]) / width_divider))
        mat['y'].append(math.floor(int(record[2]) / height_divider))
        mat['pol'].append(int(record[3]))
        if mat['x'][row-1] >= target_width or mat['y'][row-1] >= target_height:
            raise ValueError("record exceeds limits : record = {}".format(record))
                
    if mat['ts'][-1] < mat['ts'][0] > 200*1000: print("suspicious ts", mat['ts'][-1])
    
    intg = np.array([mat['x'], mat['y'], mat['ts'], mat['pol']])
    print(intg.shape)
    whole_total_events = intg.transpose()
    print(whole_total_events.shape)
    
    for i in range(math.floor(whole_total_events.shape[0]/file_len_event) + 1):
        evt_start = file_len_event * i;
        evt_end = evt_start + file_len_event;
        if evt_end >= lines: evt_end = lines - 1
        print("file {} : event {} ~ {}".format(i, evt_start, evt_end))
        
        total_events = whole_total_events[evt_start:evt_end]
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
            # https://sparse.pydata.org/en/stable/generated/sparse.COO.html#sparse.COO
            #if chunk.shape[0] > target_height * target_width * 2:
                #raise ValueError("chunk if too big {} > {}".format(chunk.shape[0], target_height * target_width * 2))
            frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'), 
                            np.ones(chunk.shape[0]).astype('int32'), 
                            (target_height, target_width, 2))   # .to_dense()
            total_frames.append(frame)
        total_frames = sparse.stack(total_frames)
        
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')    
        
        if len(total_frames) > 200*1000 / chunk_len_us: print("many frames", mat['ts'][-1])
        
        fn_dst = path_dataset_dst + '/{}_{}.pckl'.format(fn, i)
        if os.path.isfile(fn_dst): print("target already exist")
        pickle.dump(total_frames, open(fn_dst, 'wb'))