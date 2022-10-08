import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from multiprocessing import Pool
from splittings import load_split_KIC10080943

component = 'A'
y_lim = 3000 # [nHz] limit for max/min profile value
burn_in = 500
variation = ''

if component=='A':
    core_boundary = 0.117
else:
    core_boundary = 0.115

nwalkers, nsampl = 100, 2000
plot_chains = False
perturb_splittings = False
test_profile = False


modes_dir = 'KIC_10080943/modes/'+variation+component
output_dir = os.path.join('KIC_10080943', 'runs', variation, component)

def print_log(data):
    txt = str(data)
    print(txt)
    with open('LOG', 'a') as f:
        f.write(txt)
        f.write('\n')

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

MK = ModeKernel()

basis = []
mode_files = [fn for fn in os.listdir(modes_dir) if fn.startswith('mode')]
for fn in mode_files:
    if not os.path.isfile(os.path.join(modes_dir, fn)): continue
    kern_data = MK.get_kernel_and_beta(os.path.join(modes_dir, fn))
    basis.append(kern_data.kernel)
    kern = np.vstack((kern_data.r_coord, kern_data.kernel)).T
    order = kern_data.n_pg
    kernels[order] = kern
    beta[order] = kern_data.beta

n_list = list(beta.keys())
n_list.sort()
for n in n_list:
    print(n, beta[n])
#exit()

if bool(1):
    for n in kernels:
        if not -31<=n<=-21: continue
        kern = kernels[n]
        lw=1
        alph = 0.5
        if n==-21:
            lw=3
            alph = 1.0
        plt.plot(kern[:,0], kern[:,1], label=str(n), linewidth=lw, color='black', alpha=alph)
    plt.legend()
    plt.xlabel('Fractional radius')
    plt.ylabel('Rotational kernel')
    plt.show()
    exit()

G = np.vstack(basis)

svd = np.linalg.svd(G)
S = svd[1]

condi = max(S)/min(S)
print_log('condition number = %.2f'%condi)

splittings = load_split_KIC10080943(component)

if perturb_splittings:
    print_log('perturbed splittings')
    for n in splittings:
        sp = splittings[n][0]
        sp_err = splittings[n][1]
        sp += np.random.randn() * sp_err
        splittings[n] = (sp, sp_err)

#if component=='A': del(splittings[-21])

for n in splittings: print( '%i & %.4f & %.2e & %.2e & %.2e \\\\'%(n, beta[n], splittings[n][0], splittings[n][1], splittings[n][0]/beta[n]) )

N_max = len(splittings)-2
print('N_max=', N_max)

#exit()

os.makedirs(output_dir, exist_ok=True)

for N in range(2, N_max+1):

    run_output_dir = os.path.join(output_dir, str(N))
    os.makedirs(run_output_dir, exist_ok=True)

    ndim = N

    bounds = np.linspace(core_boundary, 1, N)
    bounds = list(bounds[1:-1])

    profile_r = [-0.1, core_boundary] + bounds + [1.1]

    print_log('-'*25)
    print_log('N = %i'%N)
    print_log(profile_r)

    pos = [ 2*y_lim*np.random.rand(N)-y_lim for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(splittings,), pool=pool)
        print_log('start sampling')
        sampler.run_mcmc(pos, nsampl, progress=True)

    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    MAP_est = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    for m in MAP_est: print_log(m)
    MAP = np.array(MAP_est)

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
    plt.savefig(os.path.join(run_output_dir, 'model_split.pdf'))
    plt.clf()


    xx = [i for i in range(N)]
    plt.errorbar(x=xx, y=MAP[:,0], yerr=MAP[:,1:].T, fmt='o-')
    plt.xlim(min(xx)-0.5, max(xx)+0.5)
    plt.grid()
    plt.xlabel('Zone')
    plt.ylabel('Rotation rate [nHz]')
    plt.savefig(os.path.join(run_output_dir,'profile.pdf'))
    plt.clf()

    fig = corner.corner(samples)

    fig.savefig(os.path.join(run_output_dir,'triangle.png'))
    plt.clf()










