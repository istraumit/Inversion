import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from pulse import load_pulse

MM = ['M2.7_XC0.6', 'M3.0_XC0.6', 'M3.3_XC0.6']

def load_dir(d):
    ff = [fn for fn in os.listdir(d) if fn.endswith('.mesa')]

    for fn in ff:
        path = os.path.join(d, fn)
        rr, vv = load_pulse(path, 8)
        col='white'
        if MM[0] in fn:
            col='black'
            sty = '-'
        if MM[1] in fn:
            col='black'
            sty = '--'
        if MM[2] in fn:
            col='black'
            sty = '-.'
        if col=='white': continue
        plt.plot(rr, np.sqrt(vv), sty, label=fn[6:-5], color=col)


load_dir('/home/elwood/Documents/Inversion/DATA/01_MESA/OFF_MODELS')
load_dir('/home/elwood/Documents/Inversion/DATA/01_MESA')

plt.legend()
plt.xlabel('Radius [Rstar]')
plt.ylabel('Brunt-Vaisala frequency [Hz]')
fig = plt.gcf()
fig.set_size_inches(5, 3)
plt.tight_layout()
plt.savefig('BruntV_plots/BV_diff_'+'_'.join(MM)+'.pdf')
plt.show()







