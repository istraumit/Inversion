from math import *
import numpy as np
import matplotlib.pyplot as plt
from triplets import load_triplets

day = 24 * 60 * 60
jordan = load_triplets()
COV = np.load('Jordan_covar_2.npz')['cov']

orders = list(jordan.keys())
orders.sort()

RES_P, RES_delta_P = {},{}

N_sim = 1000
for order in orders:
    if not order-1 in orders: continue
    fnp1 = jordan[order].freq[1]
    fnp1_var = jordan[order].cov[1,1]

    fn = jordan[order-1].freq[1]
    fn_var = jordan[order-1].cov[1,1]

    covar = COV[jordan[order].idx[1], jordan[order-1].idx[1]]

    M = np.array([fnp1, fn])

    C = np.zeros((2,2))
    C[0,0] = fnp1_var
    C[1,1] = fn_var
    C[0,1] = C[1,0] = covar

    delta_P_sample, Pn_sample = [],[]
    for i in range(N_sim):
        X = np.random.multivariate_normal(M, C)
        Pnp1 = 1/X[0]
        Pn   = 1/X[1]
        delta_P = day*(Pnp1 - Pn)

        Pn_sample.append(Pn)
        delta_P_sample.append(delta_P)

    RES_P[order] = (np.mean(Pn_sample), np.std(Pn_sample))
    RES_delta_P[order] = (np.mean(delta_P_sample), np.std(delta_P_sample))

orders = RES_P.keys()

xx = [RES_P[o][0] for o in orders]
xerr = np.array([RES_P[o][1] for o in orders])

yy = [RES_delta_P[o][0] for o in orders]
yerr = np.array([RES_delta_P[o][1] for o in orders])

for i,v in enumerate(orders):
    print(v, xx[i], xerr[i], yy[i], yerr[i])

plt.errorbar(xx, yy, xerr=xerr, yerr=yerr, fmt='o')
plt.xlabel('Period [day]')
plt.ylabel('$\Delta$P [sec]')
plt.show()












