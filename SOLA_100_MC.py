import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from splittings import load_splittings_Triana
from SOLA import *

component = sys.argv[1]
target_width = float(sys.argv[2])

if component=='B':
    filtr = [-i for i in range(20, 25)]
elif component=='A':
    filtr = [-i for i in range(24, 32)]
else:
    print('Bad component')
    exit()

eigenmodes = load_eigenmodes('/home/elwood/Documents/Inversion/CODE/KIC_10080943/modes/'+component, n_pg_filter=None)
K, rr, modes, betas = get_rot_basis(eigenmodes)

#K, rr = get_Fourier_basis(30)

for i in range(K.shape[0]): plt.plot(rr, K[i,:], color='blue', alpha=0.5)
plt.show()

A = get_A_matrix(K, rr)
x_center_grid = np.linspace(0.1, 0.9, 9)

for xc in x_center_grid:
    T = gaussian(rr, xc, target_width)

    c = get_c_vector(K, rr, A, T)
    #print(c)
    H = (K.T).dot(c[:-1])

    plt.plot(rr, H, label='Averaging kernel', color='blue', alpha=0.5)
    plt.plot(rr, T, label='Target', color='orange', alpha=0.5)

    plt.xlabel('r/R')
    plt.ylabel('Target function')
    plt.show()





