import sparse
import matplotlib.pyplot as plt
import pickle
import os

folder = "eswpckl"
pckls = []
for file in os.listdir("./" + folder):
    if file.endswith(".pckl"):
        pckls.append(os.path.join("./", folder, file))
print("{} pckl files found".format(len(pckls)))

for f in pckls:
    print(f)
    handle = open(f, mode='rb')
    s = pickle.load(handle)
    d = s.todense()
    fig, ax = plt.subplots(1, d.shape[0])
    fig.suptitle(f)

    for j in range(d.shape[0]):
        ax[j].imshow(d[j, :, :, 1])
    plt.show()
