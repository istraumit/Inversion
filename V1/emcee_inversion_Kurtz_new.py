import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from inversion import *
import emcee
import corner
from multiprocessing import Pool
from splittings import load_kurtz_g, load_kurtz_p_l2

plot_splittings = True

mcmc_settings = MCMCSettings()

mcmc_settings.burn_in = 500
mcmc_settings.nwalkers = 100
mcmc_settings.nsampl = 2000
mcmc_settings.n_threads = 8

y_lim = 3000
core_boundary = 0.159

modes_dir = 'KIC11145123'
output_dir = 'KIC11145123/runs'
os.makedirs(output_dir, exist_ok=True)

init_globals(core_boundary, y_lim, output_dir)

basis, beta = load_kernels(modes_dir)

SVD_check(basis)

g_sp = load_kurtz_g()
p_sp = load_kurtz_p_l2()

splittings = {**g_sp, **p_sp}

main_loop(splittings, mcmc_settings, test_for_outliers=False)







