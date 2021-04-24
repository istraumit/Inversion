import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from config import parse_conf
from Eigenmode import Eigenmode
from test_Omega import test_Omega

opt = parse_conf()
data_dir = opt['DATA_dir']

out_dir = os.path.join(data_dir, opt['splittings_dir'])
os.makedirs(out_dir, exist_ok=True)

splittings = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
orders = [int(x) for x in splittings[:,0]]
orders.sort()


omega_zero_dir = os.path.join(data_dir, opt['GYRE_stage_dir_omega_zero'])

model_path = os.path.join(omega_zero_dir, 'MODEL')
kernels = {}
for eigen_fn in os.listdir(model_path):
    if not eigen_fn.startswith('mode'): continue
    path = os.path.join(model_path, eigen_fn)
    eigenmode = Eigenmode(path)
    order = eigenmode.n_pg
    if not order in orders: continue
    kernels[order] = eigenmode.beta * eigenmode.kernel
    rr = eigenmode.r_coord

test_split = np.array([simps(test_Omega(rr)*kernels[o], rr) for o in orders])
out_path = os.path.join(data_dir, opt['splittings_dir'], 'mock_rot')
print(test_split)
np.savetxt(out_path, test_split)

plt.plot(rr, test_Omega(rr))
plt.show()




