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

data = "sun10"
samples_folder = "./raw_pic/pic_" +data+"_20/"
data_dir = samples_folder + data +"_"
samples = os.listdir(samples_folder)

sys.path.append('../logidrivepy')
controller = LogitechController()
steering_initialize = controller.steering_initialize()
logi_update = controller.logi_update()
is_connected = controller.is_connected(0)

print(f"\n---Logitech Controller Test---")
print(f"steering_initialize: {steering_initialize}")
print(f"logi_update: {logi_update}")
print(f"is_connected: {is_connected}")
if steering_initialize and logi_update and is_connected:
    print(f"All tests passed.\n")
else:
    print(f"Did not pass connection tests. HALT!\n")
    controller.steering_shutdown()
    steering_initialize = controller.steering_initialize()
    logi_update = controller.logi_update()
    is_connected = controller.is_connected(0)
    # exit  
# controller.steering_initialize()

controller.steering_shutdown()

pics=[]
for file in range(0,len(samples)): # range(0,len(samples)):
    pics.append(cv2.imread(data_dir+str(file)+".png",1))   #data_dir   samples_folder + file 
    # cv2.imshow("pic",pics[-1])
    # cv2.waitKey(1)


for i in range(0,len(pics)):
    cv2.imshow("pic",pics[i])
    cv2.waitKey(1)  #20
    steering_initialize = controller.steering_initialize()
    # if ~steering_initialize:
    #     controller.steering_shutdown()
    #     controller.steering_initialize()


    output_acc_arr = list()
    output_brk_arr = list()
    # controller.steering_initialize()
    for j in range(0,20):
        # controller.logi_update()
        # prop = controller.LogiGetStateENGINES(0).contents
        try:
            controller.logi_update()
            prop = controller.LogiGetStateENGINES(0).contents
        except:
            print("except")
            continue
        output_acc_arr.append(1-(prop.lY+pow(2,15))/pow(2,16))
        output_brk_arr.append(1-(prop.lRz+pow(2,15))/pow(2,16))
        print("{} {:.6f} {:.6f}".format(prop.lX, output_acc_arr[-1], output_brk_arr[-1]))
        time.sleep(0.001)
        # lX : wheel, lY: accel, lRz : brakes
        # acc_f.write("{:.6f}".format(1-(prop.lY+pow(2,15))/pow(2,16)))
        # acc_f.write(' ')
        # acc_f.write("{:.6f}".format(1-(prop.lRz+pow(2,15))/pow(2,16)))
        # acc_f.write('\n')
    output_pdl_np = np.array([output_acc_arr,output_brk_arr])
    with open("./output/pdl/"+data+"/"+data+"_pedal_"+str(i)+".pckl","wb") as fw:
        pickle.dump(output_pdl_np, fw)


# for i in range(0,len(pics)):
#     cv2.imshow("pic",pics[i])
#     cv2.waitKey(1)  #20
#     output_acc_arr = list()
#     output_brk_arr = list()
#     # controller.steering_initialize()
#     for j in range(0,20):
#         # controller.logi_update()
#         # prop = controller.LogiGetStateENGINES(0).contents
#         try:
#             controller.logi_update()
#             prop = controller.LogiGetStateENGINES(0).contents
#         except:
#             print("except")
#             continue
#         output_acc_arr.append(1-(prop.lY+pow(2,15))/pow(2,16))
#         output_brk_arr.append(1-(prop.lRz+pow(2,15))/pow(2,16))
#         print("{} {:.6f} {:.6f}".format(prop.lX, output_acc_arr[-1], output_brk_arr[-1]))
#         time.sleep(0.001)
#         # lX : wheel, lY: accel, lRz : brakes
#         # acc_f.write("{:.6f}".format(1-(prop.lY+pow(2,15))/pow(2,16)))
#         # acc_f.write(' ')
#         # acc_f.write("{:.6f}".format(1-(prop.lRz+pow(2,15))/pow(2,16)))
#         # acc_f.write('\n')
#     output_pdl_np = np.array([output_acc_arr,output_brk_arr])
#     with open("./output/pdl/"+data+"/"+data+"_pedal_"+str(i)+".pckl","wb") as fw:
#         pickle.dump(output_pdl_np, fw)
    


controller.steering_shutdown()