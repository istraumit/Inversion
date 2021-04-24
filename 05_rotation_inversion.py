import sys,os
from math import sqrt, exp
import numpy as np
from config import parse_conf
from utils import print_dict, load_pickle
import matplotlib.pyplot as plt
from test_Omega import test_Omega
from scipy.ndimage import gaussian_filter1d



def plot_matrix(M):
    plt.imshow(M, origin='lower')
    plt.show()

def get_cov(N, L):
    cov  = np.eye(N)
    buf = {}
    for i in range(N):
        for j in range(N):
            if i==j: continue
            d = abs(i-j)
            if d in buf:
                cov[i,j] = buf[d]
            else:
                cov[i,j] = exp(-((i-j)/L)**2)
                buf[d] = cov[i,j]
    return cov


opt = parse_conf()
data_dir = opt['DATA_dir']
rot_int_dir = opt['rotation_integrals_dir']
rot_int_path = os.path.join(data_dir, rot_int_dir)

splittings_raw = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'rotational'))
#splittings = np.loadtxt(os.path.join(data_dir, opt['splittings_dir'], 'mock_rot'))
splittings_cov = np.load(os.path.join(data_dir, opt['splittings_dir'], 'rotational_covariance.npy'))
splittings = splittings_raw[:,1]

model = load_pickle(os.path.join(rot_int_path, 'model'))

N_zones = 1000
orders = list(model[N_zones].keys())
orders.sort()
zones = list(model[N_zones][orders[0]].keys())
zones.sort()

K = np.zeros((len(orders), len(zones)))
for i,o in enumerate(orders):
    for j,z in enumerate(zones):
        K[i,j] = model[N_zones][o][z]

ls_res = np.linalg.lstsq(K, splittings, rcond=0.01)
X = ls_res[0]


U,S,V = np.linalg.svd(K)
NULL = V[len(orders):,:]

NULLsum = np.sum(NULL, axis=0)

Xg1 = X + 60*NULLsum
Xg2 = X - 100*NULLsum

gf = gaussian_filter1d
gfsig = 10

i_start = 20
rr = np.linspace(0,1,N_zones)

plt.plot(rr[i_start:], gf( Xg1[i_start:] , gfsig)  )
plt.plot(rr[i_start:], gf( Xg2[i_start:] , gfsig)  )

plt.plot(rr[i_start:], gf( X[i_start:]  , gfsig)  )

plt.xlabel('r/R')
plt.ylabel('Omega [nHz]')
plt.show()

E = K.dot(Xg1) - splittings
#print(E)
exit()


while True:
    for j in range(100):
        X = NULL.dot(2*np.random.rand(len(zones))-1)
        Xf = gaussian_filter1d(X, 20)
        plt.plot(X, color='blue', alpha=0.5)
    plt.show()

for k in range(V.shape[0]):
    plt.plot(NULL[k,:])
    plt.show()

exit()



X_prior_cov = get_cov(N_zones, 20)
X_prior_mean = np.zeros(N_zones)

while False:
    X = np.random.multivariate_normal(X_prior_mean, X_prior_cov)
    plt.plot(X)
    plt.show()

A = K
Gx = X_prior_cov
Ge = splittings_cov

Q1 = np.matmul(Gx, A.T)
Q2 = np.matmul(np.matmul(A, Gx), A.T) + Ge
Q = np.matmul(Q1, np.linalg.inv(Q2))

Y = np.random.multivariate_normal(splittings, splittings_cov)
X = np.dot(Q, Y)

XCOV = Gx - np.matmul( np.matmul(Q, A), Gx)
sigma = np.sqrt(np.diagonal(XCOV))
plot_matrix(XCOV)
rr = np.linspace(0, 1, N_zones)

plt.plot(rr, X+10*sigma)
plt.plot(rr, X-10*sigma)
plt.plot(rr, test_Omega(rr))
plt.show()










exit()

uncert = load_pickle(os.path.join(rot_int_path, 'uncert'))


N_zones = 1

orders = list(model[N_zones].keys())
orders.sort()
zones = list(model[N_zones][orders[0]].keys())
zones.sort()
L_ensamble = len(uncert[N_zones][orders[0]][0])

I_model = np.zeros((len(orders), len(zones)))
I_uncert = np.zeros((len(orders), len(zones)))
I_full = np.zeros((len(orders), len(zones), L_ensamble))

for i,o in enumerate(orders):
    for j,z in enumerate(zones):
        I_model[i,j] = model[N_zones][o][z]
        I_uncert[i,j] = np.var(uncert[N_zones][o][z])
        for k in range(L_ensamble):
            I_full[i,j,k] = uncert[N_zones][o][z][k]


def get_CHI2(Omega, model_index):
    theor_split = np.dot(I_full[:,:,model_index], Omega)
    theor_split_uncert = np.cov(np.einsum('ijk,j', I_full, Omega))

    diff = splittings - theor_split

    SIGMA = splittings_cov + theor_split_uncert
    SIGMA_inv = np.linalg.inv(SIGMA)

    CHI2 = np.dot(np.dot(SIGMA_inv, diff), diff)

    return CHI2

Om_test = 60.0*np.ones((N_zones,))

print(get_CHI2(Om_test, 0))








