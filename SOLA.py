import os,sys
import numpy as np
import matplotlib.pyplot as plt
from config import parse_conf
from Eigenmode import Eigenmode
from scipy.integrate import simps


def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * np.sqrt(2*np.pi))

def rect(xx, x0, x1, v):
    y = [ v if x0<x<x1 else 0.0 for x in xx]
    return np.array(y)

def get_rot_basis():
    opt = parse_conf()
    data_dir = opt['DATA_dir']
    omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
    model_path = os.path.join(omega_zero_dir, 'MODEL')
    basis = []
    for eigen_fn in os.listdir(model_path):
        if not eigen_fn.startswith('mode'): continue
        path = os.path.join(model_path, eigen_fn)
        eigenmode = Eigenmode(path)
        basis.append(eigenmode.kernel)
        rr = eigenmode.r_coord
    return np.array(basis), rr

def get_Fourier_basis():
    xx_f = np.linspace(0, 1, 1000)
    basis = [1+0*xx_f]
    for k in range(1, 11):
        s = np.sin(2*np.pi*k*xx_f)
        c = np.cos(2*np.pi*k*xx_f)
        basis.append(s)
        basis.append(c)
    return np.array(basis), xx_f


def plot_matrix(M):
    plt.imshow(M, origin='lower')
    plt.show()

K, rr = get_rot_basis()
#K, rr = get_Fourier_basis()
M = K.shape[0]
E = np.identity(M+1)
mu = 0.0
A = np.zeros((M+1, M+1))

for i in range(M):
    for j in range(M):
        A[i,j] = simps(K[i,:]*K[j,:], rr)

for i in range(M):
    A[i,M] = simps(K[i,:], rr)
    A[M,i] = A[i,M]

A = A + mu*E
#T = rect(rr, 0.4, 0.5, 10.0)
#x_center_grid = np.linspace(0, 0.2, 100)
#sigma_grid = np.linspace(0.001, 0.1, 100)
#qq = []
#for x_center in x_center_grid:
#for sigma in sigma_grid:
if True:
    T = gaussian(rr, 0.1, 0.01)

    v = np.zeros((M+1,))
    for i in range(M):
        v[i] = simps(K[i]*T, rr)
    v[M] = 1

    c = np.linalg.solve(A, v)

    H = (K.T).dot(c[:-1])
    quality = simps(np.abs(H-T), rr)
#    qq.append(quality)

#plt.plot(sigma_grid, qq)
#plt.show()

print('approx quality metrix = %.2e'%quality)

plt.plot(rr, H, label='Averaging kernel')
plt.plot(rr, T, label='Target')
plt.legend()
plt.xlabel('r/R')
plt.ylabel('Target function')
plt.show()













