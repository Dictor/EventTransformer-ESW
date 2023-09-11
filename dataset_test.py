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

import os
import pickle
import numpy as np
import sparse


f = open("test1_back6.pckl","rb")
data = pickle.load(f)

temp_arr = data.coords
s1 = sparse.COO.from_numpy(temp_arr)
# a = np.array([[0, 0, 0,179,179,179], [103, 157, 161,91,101,104],[1,0,1,1,0,0]])
a=np.zeros((180,240,2),np.uint8)
b=[]    # np.zeros((180,240,2),np.uint8),np.zeros((180,240,2),np.uint8)
b.append(np.zeros((180,240,2),np.uint8))
b.append(np.zeros((180,240,2),np.uint8))
a[0][103][1]=1
a[0][157][0]=1
a[0][161][1]=1
a[179][91][1]=1
a[179][101][0]=1
a[179][104][0]=1
s2= sparse.COO.from_numpy(a)
s3= sparse.COO.from_numpy(b)
# print(type(temp_arr))
# print(type(a))
print(data)
print(s2)
print(s3)
# print(a)
print(s2.coords)
# print(temp_arr)
# print(a)
# print(temp_arr[0])
# print(a[0])
# print(temp_arr.shape)
# print(a.shape)

# print(type(data))
# print(data.data)
# print(data.coords) # --> data array 얻기 가능(numpy.ndarray), [row][num],[col][num],[val][num]
# print(data.shape)
# print(data[0].data)
# print(data[0].coords) # --> data array 얻기 가능(numpy.ndarray), [row][num],[col][num],[val][num]
# print(data[0].shape)
# print(data[1].data)
# print(data[1].coords) # --> data array 얻기 가능(numpy.ndarray), [row][num],[col][num],[val][num]
# print(data[1].shape)
# print(data[2])
# print(data[4826].data)
# print(data[4826].coords) # --> data array 얻기 가능(numpy.ndarray), [row][num],[col][num],[val][num]
# print(data[4826].shape)