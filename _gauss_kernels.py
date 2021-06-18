import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from calculus import integrate

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * math.sqrt(2*math.pi))

def test_Omega(rr):
    return 100*np.sin(2*np.pi*rr)

xx = np.linspace(0, 1, 1000)
basis = [1+0*xx]
for k in range(1, 7):
    s = np.sin(2*np.pi*k*xx)
    c = np.cos(2*np.pi*k*xx)
    basis.append(s)
    basis.append(c)

test_func = gaussian(xx, 0.5, 0.1)
splits = [np.sum(b*test_func) for b in basis]

K = np.array(basis)
rec_1 = np.linalg.lstsq(K, splits, rcond=0.01)

N_zones = 5
K = []
zones = np.linspace(0.0001, 0.9999, N_zones+1)
for j,b in enumerate(basis):
    Z = []
    for zi in range(N_zones):
        I =  integrate(xx, b, zones[zi], zones[zi+1])
        Z.append(I)
    K.append(Z)

K = np.array(K)
#plt.plot(K[-1,:])
#plt.show()
#exit()


rec_2 = np.linalg.lstsq(K, splits, rcond=0.01)

plt.plot(xx, test_func, label='Original')
plt.plot([0.5*(zones[i]+zones[i+1]) for i,z in enumerate(zones[:-1])], 1e-3*rec_2[0], '.-', label='Inversion')
#plt.plot(xx, rec_1[0])

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

plt.show()





















