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

def line_zero(x, y):
    assert len(x)==2
    assert len(y)==2
    a = (y[1]-y[0])/(x[1]-x[0])
    b = y[0] - a*x[0]
    return -b/a

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
            if N20[i]>0:
                good=True
                x0 = line_zero(rr0[i-1:i+1], N20[i-1:i+1])
                rr.append(x0/R)
                NN.append(0.0)
        else:
            if N20[i]<=0:
                x0 = line_zero(rr0[i-1:i+1], N20[i-1:i+1])
                rr.append(x0/R)
                NN.append(0.0)
                break
        if good:
            rr.append(rr0[i]/R)
            NN.append(N20[i])

    rr = np.array(rr)
    NN = np.sqrt(np.array(NN))
    return rr, NN

eval_lambda_path = '/home/elwood/Soft/gyre/6.0/bin/eval_lambda'
tmp_h5 = 'TMP_LAMBDA_EVAL.H5'
def tidal_eigenvalue_lambda_single(q, l, m):
    assert abs(q)<1000.,"spin factor is too large"
    params = [l, m, q, q, 1, 'F', 'F', tmp_h5]
    par = ' '.join([str(p) for p in params])
    o = sp.check_output(eval_lambda_path + ' ' + par, shell=True, stderr=sp.STDOUT)

    fh5 = h5py.File(tmp_h5, 'r')

    lam = list(fh5['lambda'])
    assert len(lam)==1
    return lam[0]


def compute_alpha_g(r_norm, N, f_in, f_rot, l, m, n):
    n = abs(n)
    f_co = f_in - m*f_rot
    q = 2 * f_rot / f_co
    lam = tidal_eigenvalue_lambda_single(q, l, m)
    F = np.sqrt(lam) * N / (f_in - m * f_rot) / r_norm
    I = simps(F, r_norm)

    a_g = I/(2*np.pi**2) - n

    return a_g


def dfdx(opt, x):
    dx = 1.e-10
    f0 = opt(x)
    f1 = opt(x + dx)
    df = f1 - f0
    return f0, df/dx

def get_f_in_Newton(r_norm, N, f_in_start, f_rot, l, m, n, alpha_g):

    def opt(x):
        return compute_alpha_g(r_norm, N, x, f_rot, l, m, n) - alpha_g

    x0 = f_in_start
    while True:
        f0, deriv = dfdx(opt, x0)
        x1 = x0 - f0/deriv
        if abs(x0-x1)==0.0: return x1
        x0 = x1
        print(x1)

def get_f_rot_Newton(r_norm, N, f_rot_start, f_in, l, m, n, alpha_g):

    def opt(x):
        return compute_alpha_g(r_norm, N, f_in, x, l, m, n) - alpha_g

    x0 = f_rot_start
    i = 0
    max_iter = 100
    while True:
        f0, deriv = dfdx(opt, x0)
        #print('x0 =', x0, ', deriv =', deriv)
        x1 = x0 - f0/deriv
        if abs(x0-x1)<1.e-16:
            #print(abs(x0-x1))
            return x1
        x0 = x1
        i += 1
        if i>max_iter: raise Exception('Max iterations exceeded')
        #print('x1 =', x1)


def load_gyre_summary(sum_path):
    data_sum = np.loadtxt(sum_path, skiprows=6)
    S = {}
    for i in range(data_sum.shape[0]):
        l = int(data_sum[i,0])
        m = int(data_sum[i,1])
        n = int(data_sum[i,2])
        t = (l,m,n)
        S[t] = data_sum[i,-2]
    return S



if __name__=='__main__':
    pulse_path = '/home/elwood/Documents/Inversion/DATA/01_MESA/pulse_M3.0_XC0.6.mesa.omega.const'
    sum_path = 'SIMUL_sum.txt'
    data_sum = np.loadtxt(sum_path, skiprows=6)
    SM = load_gyre_summary(sum_path)
    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    plt.plot(rr, NN)
    for t in SM:
        if t[1]==0:
            #plt.axhline(SM[t])
            pass
    plt.grid()
    plt.xlabel('Radius [Rstar]')
    plt.ylabel('Brunt-Vaisala frequency [Hz]')
    plt.show()
    exit()
    f_rot = 1.e-6

    A = {}
    A[1] = {}
    A[2] = {}
    for i in range(data_sum.shape[0]):
        l = int(data_sum[i, 0])
        n = int(data_sum[i, 4])
        if n==0 or n>30: continue
        m = int(data_sum[i, 1])
        if m!=0:continue
        f_in = data_sum[i, 7]

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


#alpha_g plot as a function of radial order
if __name__=='__main__x':
    pulse_path = '/home/elwood/Documents/Inversion/DATA/01_MESA/pulse_M3.0_XC0.6.mesa.omega.const'
    sum_path = 'SIMUL_sum_m.txt'
    data_sum = np.loadtxt(sum_path, skiprows=6)

    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    f_rot = 1.e-6

    F = {0:{}, 1:{}, -1:{}}
    for i in range(data_sum.shape[0]):
        l = int(data_sum[i, 0])
        m = int(data_sum[i, 1])
        n = -int(data_sum[i, 2])
        f = data_sum[i, -2]
        F[m][n] = f

    nn = set.intersection(set(F[0].keys()), set(F[1].keys()), set(F[-1].keys()))

    AG = {0:{}, 1:{}, -1:{}}
    for n in nn:
        AG[0][n] =  compute_alpha_g(rr, NN, F[0][n], f_rot, 1, 0, n)
        AG[1][n] =  compute_alpha_g(rr, NN, F[1][n], f_rot, 1, 1, n)
        AG[-1][n] = compute_alpha_g(rr, NN, F[-1][n], f_rot, 1, -1, n)

    for m in [-1,0,1]:
        plt.plot([-n for n in nn], [AG[m][n] for n in nn], 'o-', label='m='+str(m))

    plt.legend()
    plt.xlabel('Radial order')
    plt.ylabel('Alpha_g')
    plt.show()


#rotation frequency from Eq.3 with alpha_g from m=0
if __name__=='__main__':
    pulse_path = '/home/elwood/Documents/Inversion/DATA/01_MESA/pulse_M3.0_XC0.6.mesa.omega.const'
    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    S = load_gyre_summary('SIMUL_sum_m.txt')
    Snorot = load_gyre_summary('SIMUL_sum_nonrot.txt')
    l = 1
    fstart = 1.e-8
    FROT = {1:[], -1:[]}
    nn = list(range(5, 18))
    for n in nn:
        alpha_g = compute_alpha_g(rr, NN, Snorot[(l,0,n)], 0, l, 0, n)
        m = 1
        f_rot_find = get_f_rot_Newton(rr, NN, fstart, S[(l,m,n)], l, m, n, alpha_g)
        FROT[m].append(f_rot_find*1.e9)
        m = -1
        f_rot_find = get_f_rot_Newton(rr, NN, fstart, S[(l,m,n)], l, m, n, alpha_g)
        FROT[m].append(f_rot_find*1.e9)

    for m in [-1,1]:
        plt.plot([-n for n in nn], FROT[m], 'o-', label='m='+str(m))
    plt.plot([-n for n in nn], [1.e3 for n in nn])
    plt.legend()
    plt.xlabel('Radial order')
    plt.ylabel('Rotation frequency [nHz]')
    plt.show()





