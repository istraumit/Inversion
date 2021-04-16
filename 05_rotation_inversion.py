import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from utils import condi_num, print_dict, load_pickle
import matplotlib.pyplot as plt


opt = parse_conf()
data_dir = opt['DATA_dir']
rot_int_dir = opt['rotation_integrals_dir']
rot_int_path = os.path.join(data_dir, rot_int_dir)

splittings_raw = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
splittings = {}
for i in range(splittings_raw.shape[0]):
    order = int(splittings_raw[i][0])
    splittings[order] = (splittings_raw[i][1], splittings_raw[i][2])


model = load_pickle(os.path.join(rot_int_path, 'model'))
uncert = load_pickle(os.path.join(rot_int_path, 'uncert'))



N_zones = 3

orders = list(model[N_zones].keys())
orders.sort()
zones = list(model[N_zones][orders[0]].keys())
zones.sort()
L_ensamble = len(uncert[N_zones][orders[0]][0])

K = np.zeros((len(orders), len(zones)))
KU = np.zeros((len(orders), len(zones), L_ensamble))


for i,o in enumerate(orders):
    for j,z in enumerate(zones):
        K[i,j] = model[N_zones][o][z]
        for k in range(L_ensamble):
            KU[i,j,k] = uncert[N_zones][o][z][k]

if bool(0):
    im=plt.imshow(KU, interpolation='none', origin='lower')
    plt.colorbar(im)
    plt.show()
    exit()


XX = []
for i_ens in range(L_ensamble):

    KP = KU[:,:,i_ens]
    print(np.linalg.svd(KP)[1])
    #print(condi_num(KP))

    A = np.matmul(KP.T, KP)
    
    #J = np.array([splittings[o][0] + np.random.randn()*splittings[o][1] for o in orders])
    J = np.array([splittings[o][0] for o in orders])

    b = np.dot(K.T, J)

    X = np.linalg.solve(A, b)
    XX.append(X)

for X in XX: plt.plot(X, 'o-', color='blue', alpha=0.25)
plt.show()







