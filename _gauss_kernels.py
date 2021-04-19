import os,sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def expand_fourier(xx, yy, kmax):
    xx_f = 2*np.pi*np.array(xx)
    ex = []
    for k in range(1, kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        ex.append(simps(s*yy, xx))
        ex.append(simps(c*yy, xx))
    return ex

N_inv_points = 1000
def inverse_fourier(coef, kmax):
    xx_f = np.linspace(0, 2*np.pi, N_inv_points)
    inv = 0.0*xx_f
    i = 0
    for k in range(1, kmax):
        s = np.sin(k*xx_f)
        c = np.cos(k*xx_f)
        inv += coef[i]*s
        i += 1
        inv += coef[i]*c
        i += 1
    return inv

def test_Omega(rr):
    return 100*np.sin(2*np.pi*rr)

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * math.sqrt(2*math.pi))

def rectang(xx, N, i):
    return [1.0 if (1/N)*i < x < (1/N)*(i+1) else 0.0  for x in xx]

N_r = 1000
rr = np.linspace(0,1,N_r)

M = 20
sigma = 0.5/M
kern_points = np.linspace(0, 1, M)
kernels = []
for k in range(M):
    #kernels.append(gaussian(rr, kern_points[k], sigma))
    kernels.append(rectang(rr, M, k))

J = [simps(test_Omega(rr)*kern, rr) + 0*np.random.randn() for kern in kernels]
print(J)

kmax = 51
K = np.array([expand_fourier(rr, kern, kmax) for kern in kernels])
print('------------------')
print(K.shape)
print(np.linalg.cond(K))
X,res,rank,sv = np.linalg.lstsq(K, J, rcond=0.01)
print(rank)
REC = inverse_fourier(X, kmax)
plt.plot(rr, REC, color='blue')
plt.plot(rr, test_Omega(rr))
plt.show()















