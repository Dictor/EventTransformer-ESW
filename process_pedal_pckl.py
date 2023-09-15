import pickle
import os


directory = input("please give a directory where the pedal pickles are placed (under ./)")
accs, brks = [], []

for file in os.listdir(directory):
    if file.endswith(".pckl"):
        f = open(os.path.join("./", directory , file), 'rb')
        data = pickle.load(f)
        accs.append(sum(data[0]) / len(data[0]))
        brks.append(sum(data[1]) / len(data[1]))
    
with open('result_pedal.pckl', 'wb') as f:
    pickle.dump({'accs': accs, 'brks': brks}, f, pickle.HIGHEST_PROTOCOL)