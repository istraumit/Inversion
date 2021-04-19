import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from config import parse_conf
from Eigenmode import Eigenmode
from SVD import *

def expand_fourier(xx, yy, kmax):
    xx_f = 2*np.pi*np.array(xx)
    ex = []
    for k in range(1, kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        ex.append(simps(s*yy, xx))
        ex.append(simps(c*yy, xx))
    return ex

N_inv_points = 1000
def inverse_fourier(coef, kmax):
    xx_f = np.linspace(0, 2*np.pi, N_inv_points)
    inv = 0.0*xx_f
    i = 0
    for k in range(1, kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        inv += coef[i]*s
        i += 1
        inv += coef[i]*c
        i += 1
    return inv

def test_Omega(rr):
    return 100*np.sin(2*np.pi*rr)

opt = parse_conf()
data_dir = opt['DATA_dir']
stage_dir = opt['rotation_integrals_dir']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)

splittings = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
splittings_cov = np.load(os.path.join(data_dir, opt['splittings_dir'], 'rotational_covariance.npy'))

N_split = splittings.shape[0]
N_zones_max = N_split
orders = [int(x) for x in splittings[:,0]]
orders.sort()

omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
core_boundary = float(opt['core_boundary'])
models = os.listdir(omega_zero_dir)
models.sort()

#------------------------

model_path = os.path.join(omega_zero_dir, models[0])
kernels = {}
for eigen_fn in os.listdir(model_path):
    if not eigen_fn.startswith('mode'): continue
    path = os.path.join(model_path, eigen_fn)
    eigenmode = Eigenmode(path)
    order = eigenmode.n_pg
    if not order in orders: continue
    kernels[order] = eigenmode.beta * eigenmode.kernel
    rr = eigenmode.r_coord

test_split = np.array([simps(test_Omega(rr)*kernels[o], rr) for o in orders])
print(test_split)
#------------------------

xx_rec = np.linspace(0,1,N_inv_points)

kmax = 100
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
        K.append(expand_fourier(rr, kernels[order], kmax))
    K = np.array(K)
    print(np.linalg.cond(K))

    for mc in range(1):
        split = np.array([1.0 for x in test_split]) #np.random.multivariate_normal(splittings[:,1], splittings_cov)
        X,res,rank,sv = np.linalg.lstsq(K, split, rcond=0.01)
        print(rank)
        REC = inverse_fourier(X, kmax)
        plt.plot(xx_rec, REC, color='blue', alpha=0.25)

plt.xlabel('r/R_star')
plt.ylabel('Omega [nHz]')
plt.grid()
plt.show()







