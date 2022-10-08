import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from magnetic_diff_xi import diff_xi
from inversion import ModeKernel
from bisect import bisect
import phys_const_cgs as co
from scipy.integrate import trapz
from splittings import cd_to_Hz

if bool(1):
    d = np.loadtxt('magnetic_Sc_uncert.data')
    xx = [int(d[0,2*i]) for i in range(d.shape[1]//2)]
    for i in range(d.shape[0]):
        row = []
        col = 'blue' if i<25 else 'red'
        for j in range(d.shape[1]//2):
            row.append(d[i,2*j+1])
        plt.plot(xx, row, 'o-', color=col, alpha=0.25)
    plt.xticks(xx)
    plt.xlabel('Radial order')
    plt.ylabel('Sc * omega')
    plt.show()
    exit()


def mag_b(r, rc):
    return (r/rc)**(-3)

triana_center_f = {}
with open('Triana/triana.center.freq') as f:
    for line in f:
        arr = line.split()
        triana_center_f[-int(arr[0])] = cd_to_Hz( float(arr[1]) )

core_boundary = 0.15

R_star = 2.1 * co.Rsol
M_star = 3.2 * co.Msol

C11 = 1./5

def get_Sc_om2(mode):
    rho_c = mode.rho[0]
    i_core = bisect(mode.r_coord, core_boundary)
    r_envel = mode.r_coord[i_core:]
    xi_h = mode.xi_h[i_core:]
    rho = mode.rho[i_core:]

    r_xi_h = r_envel * mag_b(r_envel, core_boundary) * xi_h
    drxihdr = diff_xi(r_envel, r_xi_h, r_envel)
    numerator = (2*drxihdr)**2
    I_numer = trapz(numerator, r_envel)
    denom = xi_h**2 * rho * r_envel**2 / rho_c
    I_denom = trapz(denom, r_envel)

    I = I_numer / I_denom

    Sc = C11*I/(8*pi * rho_c * R_star**2)
    return Sc

dbase = sys.argv[1]

subdirs = [d for d in os.listdir(dbase) if os.path.isdir(os.path.join(dbase,d))]

MK = ModeKernel()

for subdir in subdirs:
    print(subdir)
    subdir_path = os.path.join(dbase, subdir)
    mode_files = [fn for fn in os.listdir(subdir_path) if fn.startswith('mode')]
    mode_files.sort()
    kernels = {}
    SC = {}
    for mode_fn in mode_files:
        mode_path = os.path.join(subdir_path, mode_fn)
        mode = MK.get_kernel_and_beta(mode_path)

        Sc_om2 = get_Sc_om2(mode)
        SC[mode.n_pg] = Sc_om2 / triana_center_f[mode.n_pg]

    with open('magnetic_Sc_uncert.data', 'a') as f:
        for key in SC:
            f.write(str(key))
            f.write(' ')
            f.write(str(SC[key]))
            f.write(' ')
        f.write('\n')

    plt.plot(SC.keys(), SC.values(), 'o-')

plt.xlabel('Radial order')
plt.ylabel('Sc * omega')
plt.show()











