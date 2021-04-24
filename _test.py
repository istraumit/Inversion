import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from Eigenmode import Eigenmode
from calculus import integrate
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d
from SVD import *


opt = parse_conf()
data_dir = opt['DATA_dir']
stage_dir = opt['rotation_integrals_dir']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)

splittings = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
N_split = splittings.shape[0]
N_zones_max = N_split
orders = [int(x) for x in splittings[:,0]]
orders.sort()

omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
core_boundary = float(opt['core_boundary'])
models = os.listdir(omega_zero_dir)

split = [np.random.rand() for x in splittings[:,1]] #+ np.random.randn()*splittings[:,2]

def get_solution(kernels, rr):
    K = np.zeros((len(orders), len(orders)))
    for i,o1 in enumerate(orders):
        for j,o2 in enumerate(orders):
            kern_prod = kernels[o1] * kernels[o2]
            K[i,j] = simps(kern_prod, rr)

    print(K)
    input()

    print(np.linalg.cond(K))
    X,res,rank,sv = np.linalg.lstsq(K, split, rcond=0.01)
    REC = np.zeros((len(rr)))
    for i,o in enumerate(orders):
        REC += X[i]*kernels[o]

    return REC

rr_new = np.linspace(0, 1, 1000)

if bool(1):
    for model_dir in models:
        model_path = os.path.join(omega_zero_dir, model_dir)
        kernels = {}
        for eigen_fn in os.listdir(model_path):
            if not eigen_fn.startswith('mode'): continue
            path = os.path.join(model_path, eigen_fn)
            eigenmode = Eigenmode(path)
            order = eigenmode.n_pg
            if not order in orders: continue
            kernels[order] = eigenmode.beta * eigenmode.kernel
            rr = eigenmode.r_coord

        REC = get_solution(kernels, rr)
        RECi = np.interp(rr_new, rr, REC)
        plt.plot(rr_new, gaussian_filter1d(RECi, 25), color='blue', alpha=0.25)

plt.show()










