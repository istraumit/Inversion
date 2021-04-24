import os,sys
import numpy as np
import matplotlib.pyplot as plt
from config import parse_conf
from Eigenmode import Eigenmode
from SVD import truncated_SVD
from scipy.signal import lombscargle

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * np.sqrt(2*np.pi))

def get_rot_basis():
    opt = parse_conf()
    data_dir = opt['DATA_dir']
    omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])
    models = os.listdir(omega_zero_dir)
    models.sort()
    model_path = os.path.join(omega_zero_dir, models[0])
    basis = []
    for eigen_fn in os.listdir(model_path):
        if not eigen_fn.startswith('mode'): continue
        path = os.path.join(model_path, eigen_fn)
        eigenmode = Eigenmode(path)
        basis.append(eigenmode.kernel)
        rr = eigenmode.r_coord
    return np.array(basis), rr

def plot_matrix(M):
    plt.imshow(M, origin='lower')
    plt.show()

xx = np.linspace(0, 1, 1000)

kmax = 4
basis = [1+0*xx]
for k in range(1,kmax):
    s = np.sin(2*np.pi*k*xx) #* gaussian(xx, 0.5, 0.1)
    c = np.cos(2*np.pi*k*xx) #* gaussian(xx, 0.5, 0.1)
    basis.append(s)
    basis.append(c)

basis = np.array(basis)
basis, xx = get_rot_basis()

for i in range(basis.shape[0]):
    plt.plot(xx, basis[i,:])
plt.show()

freq = np.linspace(1, 3000, 10000)
for i in range(basis.shape[0]):
    PGRAM = lombscargle(xx, basis[i,:], freq, precenter=True, normalize=True)
    #PGRAM /= max(PGRAM)
    plt.plot((1/2/np.pi)*freq, PGRAM)
plt.show()
exit()


print(np.linalg.cond(basis))
G = np.matmul(basis.T, basis)
print(G.shape)
print(np.linalg.cond(G))
plot_matrix(G)

U,S,V = np.linalg.svd(G)
plot_matrix(U)
plot_matrix(V)
plt.plot(S)
plt.show()

func = gaussian(xx, 0.5, 0.1)
exp = np.matmul(basis, func)
INV = np.linalg.pinv(basis, rcond=0.01)
func_rec = np.dot(INV, exp)

func /= max(func)
func_rec /= max(func_rec)

plt.plot(xx, func)
plt.plot(xx, func_rec)
plt.show()







