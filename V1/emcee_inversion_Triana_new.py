import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from multiprocessing import Pool
from splittings import load_papics, load_Jordan_split

split_source = ['triana','papics','jordan'][1]
plot_splittings = True

mcmc_settings = MCMCSettings()

mcmc_settings.burn_in = 500
mcmc_settings.nwalkers = 100
mcmc_settings.nsampl = 2000
mcmc_settings.n_threads = 8

y_lim = 3000
core_boundary = 0.159

modes_dir = 'Triana/modes'
output_dir = 'Triana/runs/test_sim_che'
os.makedirs(output_dir, exist_ok=True)

init_globals(core_boundary, y_lim, output_dir)

basis, beta = load_kernels(modes_dir)

SVD_check(basis)

#------------------------------
def load_splittings(split_source, kind):
    splittings = {}
    if split_source=='triana':
        with open('splittings.triana') as f:
            n = 0
            for line in f:
                n += 1
                if n < 3: continue
                arr = line.split()
                splittings[-int(arr[0])] = (float(arr[2]), float(arr[4]))
    elif split_source=='papics':
        split = load_papics()
        left = True
        equal_errors = False
        error_value = 5.0
        if kind=='left':
            if equal_errors:
                for k in split:
                    splittings[k] = (split[k].left_split, error_value)
            else:
                for k in split:
                    splittings[k] = (split[k].left_split, split[k].left_split_error())
        elif kind=='right':
            if equal_errors:
                for k in split:
                    splittings[k] = (split[k].right_split, error_value)
            else:
                for k in split:
                    splittings[k] = (split[k].right_split, split[k].right_split_error())
        elif kind=='symm':
            for k in split:
                mu = 0.5*(split[k].left_split + split[k].right_split)
                sigma = max(split[k].left_split_error(), split[k].right_split_error())
                if equal_errors: sigma = error_value
                splittings[k] = (mu, sigma)
    elif split_source=='jordan':
        splittings = load_Jordan_split(kind, inflation_factor=10.0)
    else:
        print('Unknown splitting source:'+split_source)
        exit()
    return splittings

#del(splittings[-26])

#perturb_splittings(splittings)

#print_log( '%i & %.4f & %.2f & %.2e & %.2f \\\\'%(n, beta[n], splittings[n][0], splittings[n][1], splittings[n][0]/beta[n]) )

def plot_splittings(splittings, color, fmt, ms, lbl):
    if plot_splittings:
        xx = splittings.keys()
        yy = [splittings[x][0] for x in xx]
        yerr = [splittings[x][1] for x in xx]
        plt.errorbar(xx, yy, yerr=yerr, capsize=3, color=color, fmt=fmt, markersize=ms, label=lbl)

    if plot_splittings:
        plt.xlabel('Radial order')
        plt.ylabel('Frequency [nHz]')

kind = 'symm'
if kind=='symm':
    sp_triana = load_splittings('triana', kind)
    plot_splittings(sp_triana, 'black', 'x', 20, 'Triana')

sp_papics = load_splittings('papics', kind)
sp_jordan = load_splittings('jordan', kind)
plot_splittings(sp_papics, 'red', 'o', 10, 'Papics')
plot_splittings(sp_jordan, 'blue', 's', 5, 'Van Beeck')

plt.xticks(ticks=list(sp_papics.keys()))
plt.legend()
plt.show()

sim_split = {-32: (-27.06446474471389, 1.0), -31: (-27.23122427402073, 1.0), -30: (-27.3597775934683, 1.0), -29: (-26.96683215917495, 1.0), -28: (-27.055099179807698, 1.0), -27: (-26.484530893943475, 1.0), -26: (-26.138569495208024, 1.0), -25: (-26.45599677657003, 1.0), -24: (-26.016561431283314, 1.0), -23: (-26.76096656653675, 1.0), -22: (-26.8195712543031, 1.0), -21: (-26.722239281807298, 1.0), -20: (-27.185496915748814, 1.0), -19: (-26.34250087705083, 1.0), -18: (-26.65147442223635, 1.0), -17: (-26.258904422294897, 1.0), -16: (-26.01587214766977, 1.0), -15: (-27.074946231383556, 1.0), -14: (-26.971671995709862, 1.0)}


main_loop(sim_split, mcmc_settings, test_for_outliers=False)







