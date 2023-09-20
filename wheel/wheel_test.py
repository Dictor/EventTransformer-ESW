from logidrivepy import LogitechController
import sys
import time
import os
import keyboard

acc_f = open("street5_acc_test.txt",'w')
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
    exit 

controller.steering_initialize()
# for i in range(0, 20):
while(1):
    # time.sleep(0.001)
    # prop = controller.get_state_engines(0).contents
    try:
        controller.logi_update()
        prop = controller.LogiGetStateENGINES(0).contents
    except:
        print("except")
        continue
    print("{} {:.6f} {:.6f}".format(prop.lX, 1-(prop.lY+pow(2,15))/pow(2,16), 1-(prop.lRz+pow(2,15))/pow(2,16)))
    # lX : wheel, lY: accel, lRz : brakespip instal
    acc_f.write("{:.6f}".format(1-(prop.lY+pow(2,15))/pow(2,16)))
    acc_f.write(' ')
    acc_f.write("{:.6f}".format(1-(prop.lRz+pow(2,15))/pow(2,16)))
    acc_f.write('\n')
    if keyboard.is_pressed("1"):
        break

controller.steering_shutdown()
acc_f.close()
