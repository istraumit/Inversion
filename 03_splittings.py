import sys,os
from math import sqrt
import numpy as np
from config import parse_conf
from utils import cd_to_nHz
import matplotlib.pyplot as plt


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

triplets, FREQ, COV = load_triplets()
orders = list(triplets.keys())
orders.sort()

def save_splittings(splittings, name):
    with open(os.path.join(out_dir, name), 'w') as f:
        for order in orders:
            f.write(str(order)+' ')
            f.write('%.16e %.16e\n'%splittings[order])

N_mc = int(opt['splittings_N_mc'])
#-----------------------------
rotational = {}
for order in triplets:
    t = triplets[order]
    mean = 0.5*(t.freq[2] - t.freq[0])
    vari = variance(0.5, -0.5, t.cov[2,2], t.cov[0,0], t.cov[0,2])
    rotational[order] = (cd_to_nHz(mean), cd_to_nHz(sqrt(vari)))

save_splittings(rotational, 'rotational')

sample = []
for sim in range(N_mc):
    F = np.random.multivariate_normal(FREQ, COV)
    S = []
    for order in orders:
        t_idx = triplets[order].idx
        split = cd_to_nHz( 0.5*(F[t_idx[2]] - F[t_idx[0]]) )
        S.append(split)
    sample.append(S)

sample = np.array(sample)

split_mean = np.mean(sample, axis=0)
split_exact = np.array([rotational[o][0] for o in orders])
diff = np.sum(np.abs( split_mean - split_exact))
print('Absolute difference between sample mean and exact splittings:', diff)

split_cov = np.cov(sample, rowvar=False)
np.save(os.path.join(out_dir, 'rotational_covariance'), split_cov)

im=plt.imshow(np.log10(np.abs(split_cov)), origin='lower')
plt.title('log10(abs(cov)) for splittings')
plt.colorbar(im)
plt.savefig(os.path.join(out_dir, 'rot_cov.png'))
plt.clf()


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





