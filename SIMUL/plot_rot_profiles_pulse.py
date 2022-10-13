import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from pulse import load_pulse


d = '/home/elwood/Documents/Inversion/DATA/01_MESA'
ff = [fn for fn in os.listdir(d) if fn.endswith('.cheb')]

for fn in ff:
    path = os.path.join(d, fn)
    rr, vv = load_pulse(path, 18)
    vv = 0.5e9*vv/np.pi
    I = simps(vv, rr)
    print(I)
    plt.plot(rr, vv, label=fn[6:16])

plt.legend()
plt.xlabel('R [Rstar]')
plt.ylabel('Rotation frequency [nHz]')
plt.ylim(500, 1300)
plt.show()




