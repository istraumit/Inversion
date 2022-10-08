import os,sys
import numpy as np
import matplotlib.pyplot as plt
from Eigenmode import Eigenmode
from scipy.integrate import simps


def test_Omega(rr, k):
    F = {}
    F[0] = lambda x:0.0*x + 100.0
    F[1] = lambda x:100.0*x
    F[2] = lambda x:100.0 - 100.0*x
    F[3] = lambda x:100*np.sin(2*np.pi*x)
    F[4] = lambda x:100*np.sin(np.pi*x)
    F[5] = lambda x:100*np.sin(4*np.pi*x)
    return F[k](rr)

def get_simul_split(K, rr, n):
    Om = test_Omega(rr, n)
    M = K.shape[0]
    spl = []
    for i in range(M):
        s = simps(K[i,:]*Om, rr)
        spl.append((s, 0.01*s))
    return spl

def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * np.sqrt(2*np.pi))

def get_rot_basis(eigenmodes):
    basis, modes = [], []
    betas = {}
    for eigenmode in eigenmodes:
        m = (eigenmode.l, eigenmode.n_pg)
        modes.append(m)
        betas[m] = eigenmode.beta
        basis.append(eigenmode.kernel)
        rr = eigenmode.r_coord
    return np.array(basis), rr, modes, betas

def get_Fourier_basis(n):
    xx_f = np.linspace(0, 1, 1000)
    basis = [1+0*xx_f]
    for k in range(1, n):
        s = np.sin(2*np.pi*k*xx_f)
        c = np.cos(2*np.pi*k*xx_f)
        basis.append(s)
        basis.append(c)
    return np.array(basis), xx_f


def load_eigenmodes(model_path, n_pg_filter=None):
    eig = []
    for eigen_fn in os.listdir(model_path):
        if not eigen_fn.startswith('mode'): continue
        path = os.path.join(model_path, eigen_fn)
        eigenmode = Eigenmode(path)
        if n_pg_filter != None:
            if not eigenmode.n_pg in n_pg_filter: continue
        eig.append(eigenmode)
    eig = sorted(eig, key=lambda o:o.n_pg)
    return eig


def get_A_matrix(K, rr):
    M = K.shape[0]
    A = np.zeros((M+1, M+1))

    for i in range(M):
        for j in range(M):
            A[i,j] = simps(K[i,:]*K[j,:], rr)

    for i in range(M):
        A[i,M] = simps(K[i,:], rr)
        A[M,i] = A[i,M]

    return A


def get_c_vector(K, rr, A, T):
    M = K.shape[0]
    v = np.zeros((M+1,))
    for i in range(M):
        v[i] = simps(K[i]*T, rr)
    v[M] = 1
    c = np.linalg.solve(A, v)
    return c


if __name__=='__main__':
    model = 'M1.5_XC0.2'
    modes_dir = '/home/elwood/Documents/Inversion/DATA/03_GYRE_modes'

    n_pg = [-i for i in range(10, 31)]
    eig = load_eigenmodes(os.path.join(modes_dir, model), n_pg)

    K, rr, modes, betas = get_rot_basis(eig)

    A = get_A_matrix(K, rr)

    x_center_grid = np.linspace(0.1, 0.9, 9)
    target_width = float(sys.argv[1])

    for xc in x_center_grid:
        T = gaussian(rr, xc, target_width)
        c = get_c_vector(K, rr, A, T)
        H = (K.T).dot(c[:-1])

        plt.plot(rr, H, label='Averaging kernel', color='blue', alpha=0.5)
        #plt.plot(rr, T, label='Target', color='orange', alpha=0.5)
        plt.axvline(xc)

        plt.xlabel('r/R')
        plt.ylabel('Target function')
        plt.show()







