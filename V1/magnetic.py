from math import *
import numpy as np
import matplotlib.pyplot as plt
from magnetic_diff_xi import diff_xi, kernels
from bisect import bisect
from splittings import load_papics, cd_to_Hz
from Jordan import get_Jordan_triplets
from scipy.integrate import trapz
from inversion import SVD_check
import phys_const_cgs as co


def mag_b(r, rc):
    return (r/rc)**(-3)

core_boundary = 0.159


def SVD_check(basis):
    G = np.vstack(basis)
    svd = np.linalg.svd(G)
    S = svd[1]
    condi = max(S)/min(S)
    return condi


#orders = [-15,-17,-18,-19,-20,-24,-28,-29,-30]
orders = [-18,-20,-24,-28]

for N in range(2,len(orders)+1):

    M = np.zeros((N, len(orders)))
    j = 0
    for order in orders:
        mode = kernels[order]
        rho_c = mode.rho[0]
        i_core = bisect(mode.r_coord, core_boundary)
        r_envel = mode.r_coord[i_core:]
        xi_h = mode.xi_h[i_core:]
        rho = mode.rho[i_core:]

        r_xi_h = r_envel * mag_b(r_envel, core_boundary) * xi_h
        #r_xi_h = r_envel * xi_h
        drxihdr = diff_xi(r_envel, r_xi_h, r_envel)

        denom = xi_h**2 * rho * r_envel**2 / rho_c
        I_denom = trapz(denom, r_envel)

        F = 2 * drxihdr**2
        bounds = np.linspace(core_boundary, 1, N+1)
        
        for k in range(1, len(bounds)):
            i_start = bisect(r_envel, bounds[k-1])
            i_end = bisect(r_envel, bounds[k])

            r_envel_part = [bounds[k-1]] + list(r_envel[i_start:i_end]) + [bounds[k]]
            F_part = np.interp(r_envel_part, r_envel, F)

            I_part = trapz(F_part, r_envel_part)
            M[k-1,j] = I_part/I_denom
        
        j += 1
    #print(M)
    print('N =', N, 'kappa =', SVD_check(M))

