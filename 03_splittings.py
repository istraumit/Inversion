import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from utils import cd_to_nHz

def variance(a, b, var_x, var_y, var_xy):
    return a**2 * var_x + b**2 * var_y + 2*a*b*var_xy

opt = parse_conf()
data_dir = opt['DATA_dir']
stage_dir = opt['splittings_dir']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)


star_ID = opt['star_ID']
triplet_dir = os.path.join('triplets', star_ID)
sys.path.append(triplet_dir)
from triplets import load_triplets
os.chdir(triplet_dir)

triplets = load_triplets()
orders = list(triplets.keys())

def save_splittings(splittings, name):
    with open(os.path.join(out_dir, name), 'w') as f:
        for order in orders:
            f.write(str(order)+' ')
            f.write('%.16e %.16e\n'%splittings[order])

#-----------------------------
rotational = {}
for order in triplets:
    t = triplets[order]
    mean = 0.5*(t.freq[2] - t.freq[0])
    vari = variance(0.5, -0.5, t.cov[2,2], t.cov[0,0], t.cov[0,2])
    rotational[order] = (cd_to_nHz(mean), cd_to_nHz(sqrt(vari)))

save_splittings(rotational, 'rotational')

#-------------------------------
def sigma_B(freq):
    return 0.5*(freq[0] + freq[2]) - freq[1]

magnetic = {}
N_mc = 1000
for order in triplets:
    t = triplets[order]
    sample = []
    for i in range(N_mc):
        X = np.random.multivariate_normal(t.freq, t.cov)
        sig_B = cd_to_nHz(sigma_B(X))
        sample.append(sig_B)
    magnetic[order] = (cd_to_nHz(sigma_B(t.freq)), np.std(sample))

save_splittings(magnetic, 'magnetic')





