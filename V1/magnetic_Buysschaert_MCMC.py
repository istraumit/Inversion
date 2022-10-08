import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt

import emcee
import corner
from multiprocessing import Pool

data = {}
arr = np.loadtxt('magnetic_split_Jordan.data')
for i in range(arr.shape[0]):
    n = int(arr[i,0])
    if arr[i,1]<0:
        print('Excluded splitting n=', n, ' because it is negative')
    else:
        data[n] = (arr[i,1], arr[i,2])

#-----------------------
Sc_uncert_raw = {}
arr = np.loadtxt('magnetic_Sc_uncert_lowXc.data')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]//2):
        order = -int(arr[i,2*j])
        value = arr[i,2*j+1]
        if not order in Sc_uncert_raw: Sc_uncert_raw[order] = []
        Sc_uncert_raw[order].append(value)

Sc_uncert = {}
for order in Sc_uncert_raw:
    Sc_uncert[order] = np.std(Sc_uncert_raw[order])
#------------------------



Sc_omega = {}
arr = np.loadtxt('magnetic_Sc_omega.data')
for i in range(arr.shape[0]):
    n = int(arr[i,0])
    if n in data:
        Sc_omega[n] = arr[i,1]

def lnlike(B0):
    CHI2 = 0
    for order in Sc_omega:
        sig_B_theor = 1e9*Sc_omega[order] * B0**2
        theor_uncert = 1e9*Sc_uncert[order] * B0**2

        split_value = data[order][0]
        split_error = data[order][1]

        variance = split_error**2 + theor_uncert**2

        chi2_one = (sig_B_theor - split_value)**2 / variance
        CHI2 += chi2_one

    return -0.5 * CHI2

def lnprior(B0):
    global y_lim
    if 0<B0<1e4: return 0.0
    return -np.inf

def lnprob(x):
    B0 = x[0]
    lp = lnprior(B0)
    if not np.isfinite(lp):
        return -np.inf
    return np.array([lp + lnlike(B0)])

nwalkers, ndim, nsampl = 10, 1, 10000
pos = [ [ 1e4*np.random.rand()] for i in range(nwalkers)]

#    with Pool(mcmc_settings.n_threads) as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, nsampl, progress=True)

tau = sampler.get_autocorr_time()
burn_in = int(10*np.max(tau))

samples = sampler.get_chain(discard=burn_in, flat=True)

MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))

print(MAP_est)
print('Chi^2 =', -2*lnlike(MAP_est[0][0]))

B0_MAP = MAP_est[0][0]

plt.hist(samples, 100)
plt.xlabel('B0 [G]')
plt.ylabel('N samples')
plt.show()

def neg(L):
    return [-x for x in L]

orders = data.keys()
sig_B_theor = [1e9*Sc_omega[order] * B0_MAP**2 for order in orders]

theor_uncert = {}
for order in orders:
    theor_uncert[order] = 1e9*Sc_uncert[order] * B0_MAP**2

yerr = [sqrt(data[o][1]**2 + theor_uncert[o]**2) for o in orders]

plt.errorbar(neg(orders), [data[o][0] for o in orders], yerr=[data[o][1] for o in orders], fmt='o', color='black', label='Observation')
plt.errorbar(neg(orders), sig_B_theor, yerr=[theor_uncert[o] for o in orders], fmt='x', markersize=14, color='black', label='Theory')
plt.xlabel('Radial order')
plt.ylabel('Frequency shift [nHz]')
plt.legend()
plt.grid()
plt.show()














