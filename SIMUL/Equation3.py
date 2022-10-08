import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import h5py
import subprocess as sp



day = 24 * 60 * 60

def Hz_to_cd(f):
    return day * f

def cd_to_Hz(cd):
    return cd / day

def load_Brunt_Vaisala_from_pulse(path):
    with open(path) as f:
        for line in f:
            header = line.split()
            break

    M = float(header[1])
    R = float(header[2])
    L = float(header[3])

    d = np.loadtxt(path, skiprows=1)
    rr0 = d[:,1]
    N20 = d[:,8]
    rr, NN = [],[]
    good = False
    for i in range(rr0.shape[0]):
        if not good:
            if N20[i]>0: good=True
        else:
            if N20[i]<=0: break
        if good:
            rr.append(rr0[i]/R)
            NN.append(N20[i])

    rr = np.array(rr)
    NN = np.sqrt(np.array(NN))
    return rr, NN

eval_lambda_path = '/home/elwood/Soft/gyre/6.0/bin/eval_lambda'
tmp_h5 = 'FLILV'
def tidal_eigenvalue_lambda(qq, l, m):
    assert min(qq)==max(qq)
    n_q = 1

    params = [l, m, min(qq), max(qq), n_q, 'F', 'F', tmp_h5]
    par = ' '.join([str(p) for p in params])

    o = sp.check_output(eval_lambda_path + ' ' + par, shell=True, stderr=sp.STDOUT)

    fh5 = h5py.File(tmp_h5, 'r')

    lam = list(fh5['lambda'])
    assert len(lam)==1
    return lam[0]


def compute_alpha_g(r_norm, N, f_in, f_rot, l, m, n):
    Omega = f_rot
    q = 2 * Omega / f_in
    lam = tidal_eigenvalue_lambda(q, l, m)
    F = np.sqrt(lam) * N / (f_in - m * f_rot) / r_norm
    I = simps(F, r_norm)

    a_g = I/(2*np.pi**2) - n

    return a_g

def get_f_in_Newton(r_norm, N, f_in_start, f_rot, l, m, n, alpha_g):

    def opt(x):
        return compute_alpha_g(r_norm, N, x, f_rot, l, m, n) - alpha_g

    def dfdx(x):
        dx = 1.e-9
        f0 = opt(x)
        f1 = opt(x + dx)
        df = f1 - f0
        return f0, df/dx

    x0 = f_in_start
    while True:
        f0, deriv = dfdx(x0)
        x1 = x0 - f0/deriv
        if abs(x0-x1)==0.0: return x1
        x0 = x1
        print(x1)


def load_gyre_summary(sum_path):
    data_sum = np.loadtxt(sum_path, skiprows=6)
    S = {}
    for i in range(data_sum.shape[0]):
        l = int(data_sum[i,0])
        m = int(data_sum[i,1])
        n = -int(data_sum[i,2])
        t = (l,m,n)
        S[t] = data_sum[i,-2]
    return S

if __name__=='__main__':
    pulse_path = '/home/elwood/Documents/Inversion/DATA/01_MESA/pulse_M3.0_XC0.6.mesa.omega.const'
    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    f_rot = 1.e-6 + 0.0*rr

    S = load_gyre_summary('/home/elwood/Documents/Inversion/gyre_work/SIMUL_sum.txt')

    l,m,n = 2, 0, 20
    t = (l,m,n)

    f0 = cd_to_Hz(S[t])
    df = 8.e-6
    f_in_grid = np.linspace(f0-df, f0+df, 25)
    gg = []
    for f_in in f_in_grid:
        alpha_g = compute_alpha_g(rr, NN, f_in, f_rot, l, m, n)
        gg.append(alpha_g)
        #print(f_in, alpha_g)

    g0 = 0.0 #compute_alpha_g(rr, NN, f0, f_rot, l, m, n)
    fs = 0.79e-5
    f_in_find = get_f_in_Newton(rr, NN, fs, f_rot, l, m, n, g0)
    
    plt.plot(f_in_grid, gg)
    plt.axvline(fs, color='red', label='start', linestyle='--')
    plt.axvline(f_in_find, color='magenta', label='finish')
    plt.title('(l,m,-n)=' + str(t))
    plt.grid()
    plt.legend()
    plt.xlabel('f_in [Hz]')
    plt.ylabel('alpha_g')
    plt.show()


if __name__=='x__main__':
    pulse_path = '/home/elwood/Documents/Inversion/DATA/01_MESA/pulse_M3.0_XC0.6.mesa.omega.const'
    sum_path = '/home/elwood/Documents/Inversion/gyre_work/SIMUL_sum.txt'
    data_sum = np.loadtxt(sum_path, skiprows=6)

    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    f_rot = 1.e-6 + 0.0*rr

    A = {}
    A[1] = {}
    A[2] = {}
    for i in range(data_sum.shape[0]):
        l = int(data_sum[i, 0])
        n = int(data_sum[i, 4])
        if n==0 or n>30: continue
        m = int(data_sum[i, 1])
        if m!=0:continue
        f_in = cd_to_Hz(data_sum[i, 7])

        alpha_g = compute_alpha_g(rr, NN, f_in, f_rot, l, 0, n)
        A[l][n] = alpha_g
        print(l, n, alpha_g)

    plt.plot([-n for n in A[1]], [A[1][n] for n in A[1]], 'o-', label='l=1')
    plt.plot([-n for n in A[2]], [A[2][n] for n in A[2]], 'o-', label='l=2')
    plt.grid()
    plt.legend()
    plt.xlabel('Radial order')
    plt.ylabel('alpha_g')
    plt.show()














