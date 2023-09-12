from logidrivepy import LogitechController
import sys
import time
import os


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
for i in range(0, 20):
    time.sleep(0.5)
    # prop = controller.get_state_engines(0).contents
    try:
        controller.logi_update()
        prop = controller.LogiGetStateENGINES(0).contents
    except:
        print("except")
        continue
    print("{} {} {}".format(prop.lX, prop.lY, prop.lRz))
        # lX : wheel, lY: accel, lRz : brake

controller.steering_shutdown()