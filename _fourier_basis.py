import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from config import parse_conf
from Eigenmode import Eigenmode
from SVD import *

def expand_fourier(xx, yy, kmin, kmax):
    xx_f = 2*np.pi*np.array(xx) - np.pi
    ex = []
    for k in range(kmin, kmin+kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        ex.append(simps(s*yy, xx))
        ex.append(simps(c*yy, xx))
    return ex

def inverse_fourier(coef, kmin, kmax):
    xx_f = np.linspace(-np.pi, np.pi, 1000)
    inv = 0.0*xx_f
    i = 0
    for k in range(kmin, kmin+kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        inv += coef[i]*s
        i += 1
        inv += coef[i]*c
        i += 1
    return inv

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
models.sort()

kmin = int(sys.argv[1])

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

    K = []
    for order in orders:
        K.append(expand_fourier(rr, kernels[order], kmin, 6))

    K = np.array(K)
    print(np.linalg.svd(K)[1])
    QT = truncated_SVD(K, 2)
    X = np.dot(QT, splittings[:,1])
    REC = inverse_fourier(X, kmin, 6)
    plt.plot(REC)
    plt.show()
    break







