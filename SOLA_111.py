import os,sys
import numpy as np
import matplotlib.pyplot as plt
from config import parse_conf
from Eigenmode import Eigenmode
from scipy.integrate import simps
from splittings import load_kurtz_g, load_kurtz_p_l2

def test_Omega(rr):
    return 100-rr*100
    #return 0.0*rr + 100.0
    return 100*np.sin(4*np.pi*rr)

def get_simul_split(K, rr):
    Om = test_Omega(rr)
    M = K.shape[0]
    spl = []
    for i in range(M):
        s = simps(K[i,:]*Om, rr)
        spl.append(s)
    return np.array(spl)

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * np.sqrt(2*np.pi))

def rect(xx, x0, x1, v):
    y = [ v if x0<x<x1 else 0.0 for x in xx]
    return np.array(y)

def get_rot_basis(model_path):
    basis, modes = [], []
    betas = {}
    for eigen_fn in os.listdir(model_path):
        if not eigen_fn.startswith('mode'): continue
        path = os.path.join(model_path, eigen_fn)
        eigenmode = Eigenmode(path)
        if eigenmode.l in [0,6]: continue
        m = (eigenmode.l, eigenmode.n_pg)
        modes.append(m)
        betas[m] = eigenmode.beta
        basis.append(eigenmode.kernel)
        rr = eigenmode.r_coord
    return np.array(basis), rr, modes, betas

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

K, rr, modes, betas = get_rot_basis('/home/elwood/Documents/Inversion/CODE/KIC11145123')
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
x_center_grid = np.linspace(0.1, 0.9, 9)

spl_g = load_kurtz_g()
spl_p = load_kurtz_p_l2()
spl = {**spl_g, **spl_p}

for m in spl:
    spl[m] = (spl[m][0]/betas[m], spl[m][1])


splitt = []
for m in modes:
    if m in spl:
        splitt.append(spl[m][0])
    else:
        print(m)
print('Splittings:', splitt)


Omega = []
for xc in x_center_grid:
    T = gaussian(rr, xc, 0.05)

    v = np.zeros((M+1,))
    for i in range(M):
        v[i] = simps(K[i]*T, rr)
    v[M] = 1

    #print('xc = %.2f'%xc, 'cond =', np.linalg.cond(A))

    c = np.linalg.solve(A, v)
    Omega.append( np.sum(c[:-1]*np.array(splitt)) )

    H = (K.T).dot(c[:-1])
    #quality = simps(np.abs(H-T), rr)

    plt.plot(rr, H, label='Averaging kernel')
    plt.plot(rr, T, label='Target')
    plt.legend()
    plt.xlabel('r/R')
    plt.ylabel('Target function')
    plt.savefig('avg_kern/kern_xc='+'%.2f'%xc+'.pdf')
    #plt.show()
    plt.clf()



Om = test_Omega(rr)
plt.title('KIC11145123')
#plt.plot(rr, Om)
plt.plot(x_center_grid, Omega, 'o')
plt.xlabel('Radius [r/R]')
plt.ylabel('Rotation rate [nHz]')
plt.show()










