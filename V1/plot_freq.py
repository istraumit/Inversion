import numpy as np
import matplotlib.pyplot as plt

freq = []
with open('kurtz.g.freq') as f:
    for line in f:
        arr = line.split()
        if len(arr) == 0: continue
        freq.append(float(arr[1]))

for f in freq:
    plt.axvline(f)

plt.show()

