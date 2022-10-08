import sys,os
from math import sqrt, exp
import numpy as np
from config import parse_conf
from utils import print_dict, load_pickle
import matplotlib.pyplot as plt
from Eigenmode import Eigenmode
from test_Omega import test_Omega
from scipy.integrate import simps


opt = parse_conf()
data_dir = opt['DATA_dir']
models_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
model_dir = os.path.join(models_dir, 'MODEL')

kernels = {}
for eigen_fn in os.listdir(model_dir):
    if not eigen_fn.startswith('mode'): continue
    path = os.path.join(model_dir, eigen_fn)
    eigenmode = Eigenmode(path)
    order = eigenmode.n_pg
    kernels[order] = eigenmode

orders = list(kernels.keys())
orders.sort()

test_func = test_Omega(kernels[-14].r_coord)

splittings = {}
for order in orders:
    splittings[order] = (kernels[order].beta * simps(test_func*kernels[order].kernel, kernels[order].r_coord), 1.0)

print(splittings)

plt.plot(test_func)
plt.show()









