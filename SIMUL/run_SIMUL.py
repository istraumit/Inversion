import os,sys
import numpy as np
import matplotlib.pyplot as plt


def load_splittings(path, l, n_filter):
    if path.endswith('.noise'): ind = -1
    else: ind = -2
    data = np.loadtxt(path, skiprows=6)
    F = {}
    for i in range(data.shape[0]):
        if int(data[i,0]) != l: continue
        m = int(data[i,1])
        n = int(data[i,2])
        n_p = int(data[i,3])
        if n_p!=0: continue
        if not n in n_filter: continue
        if not n in F: F[n] = {}
        F[n][m] = data[i, ind]
    for n in n_filter:
        if not n in F:
            raise Exception('Mode ' + str(n) + ' is missing')
        if len(F[n])<3:
            raise Exception('Mode ' + str(n) + ' is incomplete: ' + str(F[n]))
    S = {}
    for n in F:
        S[n] = (F[n][0] - F[n][-1], F[n][1] - F[n][0])
    return S

base_dir = '/home/elwood/Documents/Inversion/DATA'
split_dir = '02_GYRE'
modes_dir = '03_GYRE_modes'
model = 'M1.5_XC0.2'
rotation = 'const'
split_fn = model + '.omega.' + rotation + '.noise'
split_path = os.path.join(base_dir, split_dir, split_fn)
n_filter = [-i for i in range(10, 31)]
l_deg_names = {1:'Dipole', 2:'Quadrupole'}

plt.subplot(211)
l_deg = 1
spl = load_splittings(split_path, l_deg, n_filter)
plt.plot([n for n in spl], [spl[n][0] for n in spl], 'o', label= l_deg_names[l_deg] + ', left')
plt.plot([n for n in spl], [spl[n][1] for n in spl], 'o', label= l_deg_names[l_deg] + ', right')
plt.legend()
plt.ylabel('Splitting [c/d]')
plt.xticks([n for n in spl if n%2==0])

plt.subplot(212)
l_deg = 2
spl = load_splittings(split_path, l_deg, n_filter)
plt.plot([n for n in spl], [spl[n][0] for n in spl], 'o', label= l_deg_names[l_deg] + ', left')
plt.plot([n for n in spl], [spl[n][1] for n in spl], 'o', label= l_deg_names[l_deg] + ', right')
plt.legend()
plt.ylabel('Splitting')
plt.xlabel('Radial order [c/d]')
plt.xticks([n for n in spl if n%2==0])
plt.show()





