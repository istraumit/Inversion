import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from utils import print_dict, load_pickle
import matplotlib.pyplot as plt


opt = parse_conf()
data_dir = opt['DATA_dir']
rot_int_dir = opt['rotation_integrals_dir']
rot_int_path = os.path.join(data_dir, rot_int_dir)

splittings_raw = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
splittings = splittings_raw[:,1]
splittings_cov = np.load(os.path.join(data_dir, opt['splittings_dir'], 'rotational_covariance.npy'))

model = load_pickle(os.path.join(rot_int_path, 'model'))
uncert = load_pickle(os.path.join(rot_int_path, 'uncert'))


N_zones = 1

orders = list(model[N_zones].keys())
orders.sort()
zones = list(model[N_zones][orders[0]].keys())
zones.sort()
L_ensamble = len(uncert[N_zones][orders[0]][0])

I_model = np.zeros((len(orders), len(zones)))
I_uncert = np.zeros((len(orders), len(zones)))
I_full = np.zeros((len(orders), len(zones), L_ensamble))

for i,o in enumerate(orders):
    for j,z in enumerate(zones):
        I_model[i,j] = model[N_zones][o][z]
        I_uncert[i,j] = np.var(uncert[N_zones][o][z])
        for k in range(L_ensamble):
            I_full[i,j,k] = uncert[N_zones][o][z][k]


def get_CHI2(Omega, model_index):
    theor_split = np.dot(I_full[:,:,model_index], Omega)
    theor_split_uncert = np.cov(np.einsum('ijk,j', I_full, Omega))

    diff = splittings - theor_split

    SIGMA = splittings_cov + theor_split_uncert
    SIGMA_inv = np.linalg.inv(SIGMA)

    CHI2 = np.dot(np.dot(SIGMA_inv, diff), diff)

    return CHI2

Om_test = 60.0*np.ones((N_zones,))

print(get_CHI2(Om_test, 0))








