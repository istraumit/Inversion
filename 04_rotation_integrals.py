import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from Eigenmode import Eigenmode
from calculus import integrate
from utils import condi_num
import pickle


opt = parse_conf()
data_dir = opt['DATA_dir']
stage_dir = opt['rotation_integrals_dir']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)

splittings = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
N_split = splittings.shape[0]
N_zones_max = N_split - 2
orders = [int(x) for x in splittings[:,0]]

omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
core_boundary = float(opt['core_boundary'])

U,M = {},{}
#N_zones_max = 2 # TEST
for N_zones in range(1, N_zones_max+1):
    U[N_zones] = {}
    M[N_zones] = {}
    zones = np.linspace(core_boundary, 0.9999, N_zones+1)

    for model_dir in os.listdir(omega_zero_dir):
        
        model_path = os.path.join(omega_zero_dir, model_dir)
        for eigen_fn in os.listdir(model_path):
            if not eigen_fn.startswith('mode'): continue
            path = os.path.join(model_path, eigen_fn)
            eigenmode = Eigenmode(path)
            order = eigenmode.n_pg
            if not order in orders: continue
            if not order in U[N_zones]:
                U[N_zones][order] = {}
                M[N_zones][order] = {}

            for zi in range(N_zones):
                if not zi in U[N_zones][order]: U[N_zones][order][zi] = []
                I = eigenmode.beta * integrate(eigenmode.r_coord, eigenmode.kernel, zones[zi], zones[zi+1])
                U[N_zones][order][zi].append(I)
                if model_dir == opt['best_model']: M[N_zones][order][zi] = I


UA = {}
for N_zones in U:
    UA[N_zones] = {}
    K = np.zeros((len(orders), N_zones))
    for i,order in enumerate(orders):
        UA[N_zones][order] = {}
        zis = list(U[N_zones][order].keys())
        zis.sort()
        for j,zi in enumerate(zis):
            UA[N_zones][order][zi] = np.std(U[N_zones][order][zi])
            K[i,j] = M[N_zones][order][zi]

with open(os.path.join(out_dir, 'model'), 'wb') as f:
    pickle.dump(M, f)
with open(os.path.join(out_dir, 'uncert'), 'wb') as f:
    pickle.dump(UA, f)



