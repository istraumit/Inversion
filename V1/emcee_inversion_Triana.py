import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from splittings import *

y_lim = 3000 # [nHz] limit for max/min profile value
burn_in = 2
core_boundary = 0.159
N = 9
plot_chains = False

bounds = np.linspace(core_boundary, 1, N)
bounds = list(bounds[1:-1])

profile_r = [-0.1, 0.159] + bounds + [1.1]

def print_log(data):
    txt = str(data)
    print(txt)
    with open('LOG', 'a') as f:
        f.write(txt)
        f.write('\n')

print_log('-'*25)
print_log('N = %i'%N)
print_log(profile_r)

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

    prof_y = np.piecewise(kern_x, bool_list, profile)

    Q = prof_y * kern_R

    I = np.trapz(Q, kern_x)
    return I

def lnlike(x, split):
    CHI2 = 0
    for order in split:
        I = beta[order] * get_integral(kernels[order], x)
        split_value = split[order][0]
        split_error = split[order][1]
        chi2_one = ((I - split_value) / split_error)**2
        CHI2 += chi2_one

    return -0.5 * CHI2

def lnprior(x):
    b = all([-y_lim < _ < y_lim for _ in x])
    if b: return 0.0
    return -np.inf

def lnprob(x, data):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return np.array([lp + lnlike(x, data)])

kernels = {}
beta = {}

if bool(0): # 1-kernels / 0-modes -----------------------------------
    kernels_dir = 'rot_kernels'
    kernel_files = os.listdir(kernels_dir)
    for fn in kernel_files:
        order = int(fn[12:15])
        kernels[order] = np.loadtxt(os.path.join(kernels_dir, fn))

    with open('beta') as f:
        for line in f:
            arr = line.split()
            n_pg = int(arr[0])
            bt = float(arr[1])
            beta[n_pg] = bt
else:
    modes_dir = 'modes_615_nHz'
    MK = ModeKernel()

    mode_files = os.listdir(modes_dir)
    for fn in mode_files:
        kern_data = MK.get_kernel_and_beta(os.path.join(modes_dir, fn))
        kern = np.vstack((kern_data.r_coord, kern_data.kernel)).T
        order = abs(kern_data.n_pg)
        kernels[order] = kern
        beta[order] = kern_data.beta





kern_x = kernels[20][:,0]
bool_list = []
for i in range(1, len(profile_r)):
    bool_list.append( (profile_r[i-1] <= kern_x) & (kern_x < profile_r[i]) )

if bool(0):
    for n in [14, 32]:
        plt.plot(kernels[n][:,0], kernels[n][:,1])
    plt.show()
    exit()







splittings = {}
if bool(0): # 1-triana / 0-papics -----------------------------
    with open('splittings.triana') as f:
        n = 0
        for line in f:
            n += 1
            if n < 3: continue
            arr = line.split()
            splittings[int(arr[0])] = (float(arr[2]), float(arr[4]))
else:
    split = load_papics()
    left = False
    equal_errors = True
    error_value = 5.0
    if left:
        if equal_errors:
            for k in split:
                splittings[k] = (split[k].left_split, error_value)
        else:
            for k in split:
                splittings[k] = (split[k].left_split, split[k].left_split_error())
    else:
        if equal_errors:
            for k in split:
                splittings[k] = (split[k].right_split, error_value)
        else:
            for k in split:
                splittings[k] = (split[k].right_split, split[k].right_split_error())






ndim, nwalkers, nsampl = N, 100, 1000
pos = [ 2*y_lim*np.random.rand(N)-y_lim for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(splittings,))
print_log('start sampling')
sampler.run_mcmc(pos, nsampl)

if plot_chains:
    print(sampler.chain.shape)
    for i in range(N):
        plt.plot(sampler.chain[nwalkers//2, :, i], label='zone '+str(i))

    plt.legend()
    plt.xlabel('Iteration')
    plt.show()

print('acceptance fractions:')
print(sampler.acceptance_fraction)
samples = sampler.chain[:, nsampl//burn_in:, :].reshape((-1, ndim))
#print('MCMC result:')
MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
for m in MAP_est: print_log(m)
MAP = np.array(MAP_est)

if bool(1):
    prof = MAP[:,0]

    CHI2 = -2*lnlike(prof, splittings)
    print_log('CHI^2 = %.1f'%CHI2)

    model_split = {}
    xx,yy_data,yy_data_err, yy_model = [],[],[],[]
    for order in splittings:
        model_split[order] = beta[order] * get_integral(kernels[order], prof)
        xx.append(order)
        yy_model.append(model_split[order])
        yy_data.append(splittings[order][0])
        yy_data_err.append(splittings[order][1])
    
    plt.plot(xx, yy_model, 's', label='model')
    plt.errorbar(xx, yy_data, fmt='o', yerr=yy_data_err, label='data')
    plt.xlim(min(xx)-0.5, max(xx)+0.5)
    plt.legend()
    plt.title('Chi^2 = %.2f'%CHI2)
    plt.xlabel('Radial order')
    plt.ylabel('Splitting [nHz]')
    plt.savefig('model_split.pdf')
    plt.clf()

xx = [i for i in range(N)]
plt.errorbar(x=xx, y=MAP[:,0], yerr=MAP[:,1:].T, fmt='o')
plt.xlim(min(xx)-0.5, max(xx)+0.5)
plt.grid()
plt.xlabel('Zone')
plt.ylabel('Rotation rate [nHz]')
plt.savefig('profile.pdf')
plt.clf()

fig = corner.corner(samples)

fig.savefig('triangle.png')











