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

def sigma_B(freq):
    return 0.5*(freq[0] + freq[2]) - freq[1]

N_mc = 1000
jordan = get_Jordan_triplets()
if bool(0):
    for n in jordan:
        t = jordan[n].jordan_triplet
        if t==None: continue
        print('-'*25)
        print('n =', n)
        print(t.freq)
        print(t.cov)
    exit()

jordan_sig_B = {}
for order in jordan:
    t = jordan[order].jordan_triplet
    if t==None: continue
    sample = []
    for i in range(N_mc):
        X = np.random.multivariate_normal(t.freq, t.cov)
        sig_B = cd_to_Hz(sigma_B(X))
        sample.append(sig_B)
    jordan_sig_B[-order] = (np.mean(sample), np.std(sample))
    print(order, 1e9*cd_to_Hz( sigma_B(t.freq) ), 1e9*np.std(sample))

#exit()
orders = list(jordan_sig_B.keys())
orders.sort()



core_boundary = 0.15

R_star = 2.1 * co.Rsol
M_star = 3.2 * co.Msol

C11 = 1./5

print('Order    I     Sc[G^-2]   B[G]')
print('-'*30)
Sc_omega = {}
for order in orders:
    t = jordan[-order].jordan_triplet
    omega = cd_to_Hz( t.freq[1] )
    sig_B = cd_to_Hz( sigma_B(t.freq) )

    mode = kernels[order]

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

    Sc = C11*I/(8*pi * omega**2 * rho_c * R_star**2)

    B0_square = sig_B/(omega*Sc)
    B0 = '%.0f'%sqrt(B0_square) if B0_square>0 else '--'

    print(order, '|', '%.2e'%I, '|', '%.2e'%Sc, '|', B0)
    Sc_omega[order] = Sc*omega

with open('magnetic_Sc_omega.data', 'w') as f:
    for order in Sc_omega:
        f.write(str(-order) + ' ' + str(Sc_omega[order]) + '\n')

exit()
print('-'*30)
B0_avg = 6078
sig_obs, sig_theor = [],[]
for order in orders:
    sig_B_theor = Sc_omega[order] * B0_avg**2
    sig_theor.append(sig_B_theor)

plt.errorbar(orders, [jordan_sig_B[o][0] for o in orders], yerr=[20*jordan_sig_B[o][1] for o in orders], fmt='o', color='black', label='Observation')
plt.plot(orders, sig_theor, 'x', markersize=16, color='black', label='Theory')
plt.xlabel('Radial order')
plt.ylabel('Frequency shift [Hz]')
plt.legend()
plt.grid()
plt.show()

