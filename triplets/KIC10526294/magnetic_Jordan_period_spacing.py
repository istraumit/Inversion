from math import *
import numpy as np
import matplotlib.pyplot as plt
from triplets import load_triplets
from P_spacings_covariance import get_P_spacings_with_covariance

day = 24 * 60 * 60
jordan = load_triplets()
COV = np.load('Jordan_covar_2.npz')['cov']

orders = list(jordan.keys())
orders.sort()
N = len(orders)

F = np.zeros((N,))
FCOV = np.zeros((N,N))

for i,order1 in enumerate(orders):
    F[i] = jordan[order1].freq[1]
    for j,order2 in enumerate(orders):
        FCOV[i,j] = COV[jordan[order1].idx[1], jordan[order2].idx[1]]

print(F)

im=plt.imshow(FCOV, interpolation=None, origin='lower')
plt.title('Frequency covariance')
plt.colorbar(im)
plt.show()

pairs = []
for i,order in enumerate(orders):
    if order-1 in orders: pairs.append((i-1,i))

PSP,PCOV = get_P_spacings_with_covariance(F, FCOV, pairs, N_MC_samples=100)
print(day*PSP)

im=plt.imshow(PCOV, interpolation=None, origin='lower')
plt.colorbar(im)
plt.title('Period spacing covariance')
plt.show()












