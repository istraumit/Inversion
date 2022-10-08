import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from splittings import load_kurtz_g, load_kurtz_p_l2

y_lim = 3000 # [nHz] limit for max/min profile value
burn_in = 500
core_boundary = 0.048
N = 2
plot_chains = False
perturb_splittings = False
test_profile = False
ndim, nwalkers, nsampl = N, 100, 1000

bounds = np.linspace(core_boundary, 1, N)
bounds = list(bounds[1:-1])

profile_r = [-0.1, core_boundary] + bounds + [1.1]

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

    bool_list = []
    for i in range(1, len(profile_r)):
        bool_list.append( (profile_r[i-1] <= kern_x) & (kern_x < profile_r[i]) )

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

modes_dir = 'KIC11145123'
MK = ModeKernel()

basis = []
mode_files = os.listdir(modes_dir)
for fn in mode_files:
    if not os.path.isfile(os.path.join(modes_dir, fn)): continue
    kern_data = MK.get_kernel_and_beta(os.path.join(modes_dir, fn))
    if not kern_data.l in [1,2]: continue
    if kern_data.n_pg < -15:
        basis.append(kern_data.kernel)
    kern = np.vstack((kern_data.r_coord, kern_data.kernel)).T
    order = 100*kern_data.l + kern_data.n_pg
    kernels[order] = kern
    beta[order] = kern_data.beta


G = np.vstack(basis)

svd = np.linalg.svd(G)
S = svd[1]

condi = max(S)/min(S)
print_log('condition number = %.2f'%condi)



g_sp = load_kurtz_g()
p_sp = load_kurtz_p_l2()

splittings = {**g_sp, **p_sp}

if perturb_splittings:
    print_log('perturbed splittings')
    for n in splittings:
        sp = splittings[n][0]
        sp_err = splittings[n][1]
        sp += np.random.randn() * sp_err
        splittings[n] = (sp, sp_err)

for n in splittings: print( '%i & %.4f & %.2f & %.2e & %.2f \\\\'%(n, beta[n], splittings[n][0], splittings[n][1], splittings[n][0]/beta[n]) )


if not test_profile:

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
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    for m in MAP_est: print_log(m)
    MAP = np.array(MAP_est)

if bool(1):
    if test_profile:
        prof = np.array([692, 109])
    else:
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
    plt.title('Chi^2 = %.2f'%CHI2)
    plt.xlabel('Splitting ID')
    plt.ylabel('Splitting [nHz]')
    plt.savefig('model_split.pdf')
    plt.clf()


if not test_profile:
    xx = [i for i in range(N)]
    plt.errorbar(x=xx, y=MAP[:,0], yerr=MAP[:,1:].T, fmt='o-')
    plt.xlim(min(xx)-0.5, max(xx)+0.5)
    plt.grid()
    plt.xlabel('Zone')
    plt.ylabel('Rotation rate [nHz]')
    plt.savefig('profile.pdf')
    plt.clf()

    fig = corner.corner(samples)

    fig.savefig('triangle.png')











