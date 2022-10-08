import os,sys
import numpy as np
import matplotlib.pyplot as plt
from config import parse_conf
from Eigenmode import Eigenmode
from scipy.integrate import simps
from splittings import load_splittings_Triana
from SOLA import *

test_index = int(sys.argv[1])
#K, rr, modes, betas = get_rot_basis('/home/elwood/Documents/Inversion/CODE/Triana/modes')
K, rr, modes, betas = get_rot_basis('/home/elwood/Documents/Inversion/DATA/02_GYRE_omega_zero/MODEL')

A = get_A_matrix(K, rr)

x_center_grid = np.linspace(0.1, 0.9, 9)

spl = load_splittings_Triana()

for m in spl:
    spl[m] = (spl[m][0]/betas[m], spl[m][1]/betas[m])

splitt = []
for m in modes:
    if m in spl:
        splitt.append(spl[m])

if test_index>=0:
    splitt = get_simul_split(K, rr, test_index)

N_mc = 100
Omega = []

for mc in range(N_mc):
    spl_mc = []
    for sp in splitt: spl_mc.append(sp[0] + np.random.randn()*sp[1])
    om = []
    for xc in x_center_grid:
        T = gaussian(rr, xc, 0.05)

        c = get_c_vector(K, rr, A, T)

        om.append( np.sum(c[:-1]*spl_mc) )

        if mc==0:
            H = (K.T).dot(c[:-1])

            plt.plot(rr, H, label='Averaging kernel')
            plt.plot(rr, T, label='Target')
            plt.xlabel('r/R')
            plt.ylabel('Target function')
            plt.savefig('avg_kern/kern_xc='+'%.2f'%xc+'.pdf')
            plt.clf()


    Omega.append(om)

Omega = np.array(Omega)

om_mean = np.mean(Omega, axis=0)
om_std = np.std(Omega, axis=0)

#for i,xc in enumerate(x_center_grid): print(xc, om_mean[i], om_std[i])


#plt.title('KIC10526294')
plt.rcParams.update({'font.size': 14})
plt.errorbar(x_center_grid, om_mean, yerr=om_std, fmt='o')
if test_index>=0:
    plt.plot(rr, test_Omega(rr, test_index))
plt.xlabel('Radius [r/R]')
plt.ylabel('Rotation rate [nHz]')
plt.show()










