import cv2
import os
from logidrivepy import LogitechController
import sys
import keyboard
import sparse
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.image as img

data = "back6"
samples_folder = "./raw_pic/pic_" +data+"_20/"
samples_acc_folder = "./output/pdl/"+data+"/"
data_dir = samples_folder + data +"_"
acc_dir = samples_acc_folder + data +"_pedal_"
samples = os.listdir(samples_folder)
samples_acc =  os.listdir(samples_acc_folder)
    
pics=[]
accs=[]
for file in range(0,len(samples)): # range(0,len(samples)):
    pics.append(cv2.imread(data_dir+str(file)+".png",1))   #data_dir   samples_folder + file 
    with open(acc_dir+str(file)+".pckl","rb") as fr:
        accs.append(pickle.load(fr))
    # cv2.imshow("pic",pics[-1])
    # cv2.waitKey(1)
for i in range(0,len(pics)):
    cv2.imshow("pic",pics[i])
    cv2.waitKey(20)  #20
    for j in range(0,20):
        print("acc "+str(i)+" "+str(j)+" pckl: " + str(accs[i][0][j]))
        print("brk "+str(i)+" "+str(j)+" pckl: " + str(accs[i][1][j]))
    