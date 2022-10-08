import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from multiprocessing import Pool
from splittings import load_split_KIC9751996

mcmc_settings = MCMCSettings()

mcmc_settings.burn_in = 500
mcmc_settings.nwalkers = 100
mcmc_settings.nsampl = 2000
mcmc_settings.n_threads = 3

y_lim = 3000
core_boundary = 0.046

modes_dir = 'KIC_9751996'
output_dir = os.path.join(modes_dir, 'runs_9283749823')
os.makedirs(output_dir, exist_ok=True)

init_globals(core_boundary, y_lim, output_dir)

basis, beta = load_kernels(modes_dir)

SVD_check(basis)

splittings_ = load_split_KIC9751996()

#perturb_splittings(splittings_)

for n in splittings_: print_log( '%i & %.4f & %.2f & %.2e & %.2f \\\\'%(n, beta[n], splittings_[n][0], splittings_[n][1], splittings_[n][0]/beta[n]) )


#main_loop(splittings_, mcmc_settings)







