import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
from bisect import bisect
from numpy.polynomial.chebyshev import chebval

import emcee
import corner
from multiprocessing import Pool


def init_globals(_core_boundary, _y_lim, _output_dir):
    global core_boundary, y_lim, output_dir
    y_lim = _y_lim # [nHz] limit for max/min profile value
    output_dir = _output_dir
    core_boundary = _core_boundary



class ModeRotationalData:

    def __init__(self):
        self.l = None
        self.n_pg = None
        self.r_coord = None
        self.kernel = None
        self.beta = None


class ModeKernel:

    def get_kernel_and_beta(self, fn):
        with open(fn) as f:
            n = 0
            for line in f:
                n += 1
                if n==3: header = line.split()
                if n==4:
                    values = line.split()
                    break
        res = ModeRotationalData()
        res.l = int(values[header.index('l')])
        res.n_pg = int(values[header.index('n_pg')])

        data = np.loadtxt(fn, skiprows=6)
        xiR, xiH, rho, r = data[:,0], data[:,2], data[:,4], data[:,5]

        L2 = res.l*(res.l + 1)
        R = (xiR**2 + L2 * xiH**2 - 2*xiR*xiH - xiH**2) * r**2 * rho
        R_int = simps(R, r)
        res.kernel = R / R_int

        Q = (xiR**2 + L2 * xiH**2) * r**2 * rho
        Q_int = simps(Q, r)
        res.beta = R_int / Q_int
        res.r_coord = r
        res.xi_r = xiR
        res.xi_h = xiH
        res.rho = rho

        return res


def load_kernels(modes_dir):
    global kernels, beta
    kernels = {}
    beta = {}

    MK = ModeKernel()

    basis = []
    mode_files = os.listdir(modes_dir)
    for fn in mode_files:
        if not os.path.isfile(os.path.join(modes_dir, fn)): continue
        kern_data = MK.get_kernel_and_beta(os.path.join(modes_dir, fn))
        basis.append(kern_data.kernel)
        kern = np.vstack((kern_data.r_coord, kern_data.kernel)).T
        order = kern_data.n_pg
        kernels[order] = kern
        beta[order] = kern_data.beta
    return basis, beta


#----------------------------------------------

def SVD_check(basis):
    G = np.vstack(basis)
    svd = np.linalg.svd(G)
    S = svd[1]
    condi = max(S)/min(S)
    print_log('condition number = %.2f'%condi)

#----------------------------------------------

def perturb_splittings(splittings_):
    print_log('perturbed splittings')
    for n in splittings_:
        sp = splittings_[n][0]
        sp_err = splittings_[n][1]
        sp += np.random.randn() * sp_err
        splittings_[n] = (sp, sp_err)

#----------------------------------------------

def get_integral_linear(kernel, profile):
    kern_x = kernel[:,0]
    kern_R = kernel[:,1]
    xx = np.linspace(min(kern_x), max(kern_x), len(profile))
    prof_inter = np.interp(kern_x, xx, profile)
    Q = prof_inter * kern_R
    I = np.trapz(Q, kern_x)
    return I

def get_integral(kernel, profile):
    kern_x = kernel[:,0]
    kern_R = kernel[:,1]

    bool_list = []
    for i in range(1, len(profile_r)):
        bool_list.append( (profile_r[i-1] <= kern_x) & (kern_x < profile_r[i]) )

    prof_y = np.piecewise(kern_x, bool_list, profile)

    Q = prof_y * kern_R

    I = np.trapz(Q, kern_x)
    return I

def get_integral_cheb(kernel, Cheb_coefs):
    kern_x = kernel[:,0]
    kern_R = kernel[:,1]
    
    Cheb_x = 2*(kern_x - kern_x[0])/(kern_x[1] - kern_x[0]) - 1
    Cheb_poly = chebval(Cheb_x, Cheb_coefs)

    Q = kern_R * Cheb_poly
    I = np.trapz(Q, kern_x)
    return I


def lnlike(x, split):
    CHI2 = 0
    for order in split:
        I = beta[order] * get_integral_cheb(kernels[order], x)
        split_value = split[order][0]
        split_error = split[order][1]
        chi2_one = ((I - split_value) / split_error)**2
        CHI2 += chi2_one

    return -0.5 * CHI2

def lnprior(x):
    global y_lim
    b = all([-y_lim < _ < y_lim for _ in x])
    if b: return 0.0
    return -np.inf

def lnprob(x, data):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return np.array([lp + lnlike(x, data)])

def print_log(data):
    txt = str(data)
    print(txt)
    log_dir = os.path.join(output_dir, 'LOG')
    with open(log_dir, 'a') as f:
        f.write(txt)
        f.write('\n')

def save_plot(fn):
    fig = plt.gcf()
    fig.set_size_inches(6.4, 4.8)
    fig.savefig(fn)
    fig.clf()



def save_run_results(MAP, splittings, run_output_dir):
    os.makedirs(run_output_dir, exist_ok=True)
    prof = MAP[:,0]

    CHI2 = -2*lnlike(prof, splittings)
    print_log('CHI^2 = %.1f'%CHI2)

    model_split = {}
    xx,yy_data,yy_data_err, yy_model = [],[],[],[]
    i=1
    for order in splittings:
        model_split[order] = beta[order] * get_integral(kernels[order], prof)
        xx.append(i)
        i += 1
        yy_model.append(model_split[order])
        yy_data.append(splittings[order][0])
        yy_data_err.append(splittings[order][1])


    plt.plot(xx, yy_model, 's', label='model')
    plt.errorbar(xx, yy_data, fmt='o', yerr=yy_data_err, label='data')
    plt.xlim(min(xx)-0.5, max(xx)+0.5)
    #plt.legend()
    #plt.title('Chi^2 = %.2f'%CHI2)
    plt.xlabel('Splitting ID')
    plt.ylabel('Splitting [nHz]')
    save_plot(os.path.join(run_output_dir, 'model_split.pdf'))


    xx = [i for i in range(MAP.shape[0])]
    plt.errorbar(x=xx, y=MAP[:,0], yerr=MAP[:,1:].T, fmt='o-')
    plt.xlim(min(xx)-0.5, max(xx)+0.5)
    plt.grid()
    plt.xlabel('Zone')
    plt.ylabel('Rotation rate [nHz]')
    save_plot(os.path.join(run_output_dir,'profile.pdf'))

    #fig = corner.corner(samples)
    #fig.savefig(os.path.join(run_output_dir,'triangle.png'))
    #plt.clf()



class MCMCSettings:
    pass


def run_mcmc(N, remove, splittings_, mcmc_settings):

    burn_in = mcmc_settings.burn_in
    nwalkers, nsampl = mcmc_settings.nwalkers, mcmc_settings.nsampl


    splittings = dict(splittings_)
    if str(remove)!='none':
        del(splittings[remove])


    ndim = N

    bounds = np.linspace(core_boundary, 1, N)
    bounds = list(bounds[1:-1])

    global profile_r
    profile_r = [-0.1, core_boundary] + bounds + [1.1]

    print_log('-'*25)
    print_log('N = %i'%N)
    print_log('remove = ' + str(remove))
    print_log(profile_r)

    pos = [ 2*y_lim*np.random.rand(N)-y_lim for i in range(nwalkers)]

    with Pool(mcmc_settings.n_threads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(splittings,), pool=pool)
        print_log('start sampling')
        sampler.run_mcmc(pos, nsampl, progress=True)

    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    for m in MAP_est: print_log(m)
    MAP = np.array(MAP_est)

    run_output_dir = os.path.join(output_dir, str(N), str(remove))
    save_run_results(MAP, splittings, run_output_dir)
    np.save(os.path.join(run_output_dir,'samples.npy'), samples)



def main_loop(splittings, mcmc_settings, test_for_outliers=True):

    removal_list = ['none']
    if test_for_outliers:
        lsp = list(splittings)
        lsp.sort()
        removal_list.extend(lsp)

    for remove in removal_list:
        if str(remove)=='none':
            N_max = len(splittings) - 2
        else:
            N_max = len(splittings) - 3

        for N in [6]: # range(2, N_max + 1):
            run_mcmc(N, remove, splittings, mcmc_settings)






if __name__ == '__main__':
    dirr = 'KIC11145123'
    mk = ModeKernel()
    for fn in os.listdir(dirr):
        if not os.path.isfile(os.path.join(dirr, fn)): continue
        data = mk.get_kernel_and_beta(os.path.join(dirr, fn))
    
        if data.l != 6 and data.n_pg >= -19:
            lbl = 'l=%i, n=%i'%(data.l, data.n_pg)
            print(lbl)
            plt.plot(data.r_coord, data.kernel, label=lbl)

        if data.l==2 and data.n_pg == 3:
            r_c = 0.048
            i_core = bisect(data.r_coord, r_c)
            core = data.kernel[:i_core]
            envel = data.kernel[i_core:]
            I_core = trapz(core)
            I_env = trapz(envel)
            I_total = trapz(data.kernel)
            print(I_core/I_total, I_env/I_total)    
            

    plt.axvline(0.048, color='black', linewidth=2)
    plt.legend()
    #plt.grid()
    plt.xlabel('Radius [R_star]')
    plt.show()   












