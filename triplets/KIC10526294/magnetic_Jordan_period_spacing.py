from math import *
import numpy as np
import matplotlib.pyplot as plt
from triplets import load_triplets
from P_spacings_covariance import get_P_spacings_with_covariance

day = 24 * 60 * 60

def cd_to_Hz(cd):
    return cd / day

def cd_to_uHz(f):
    return 1.e6*cd_to_Hz(f)

jordan,_,_ = load_triplets()
COV = np.load('Jordan_covar_2.npz')['cov']

orders = list(jordan.keys())
orders.sort()
N = len(orders)

F = np.zeros((N,))
FCOV = np.zeros((N,N))

for i,order1 in enumerate(orders):
    F[i] = cd_to_uHz(jordan[order1].freq[1])
    for j,order2 in enumerate(orders):
        sig2 = COV[jordan[order1].idx[1], jordan[order2].idx[1]]
        FCOV[i,j] =  sig2 * cd_to_uHz(1.0)**2

print(orders)
print(F)

with open('Jordan_freq.data', 'w') as f:
    for i,order in enumerate(orders):
        s = '%i %.8f\n'%(order, F[i])
        f.write(s)

np.savetxt('Jordan_cov.data', FCOV)

if bool(1):
    im=plt.imshow(FCOV, interpolation=None, origin='lower')
    plt.title('Frequency covariance')
    plt.colorbar(im)
    plt.show()
    exit()

pairs = []
for i,order in enumerate(orders):
    if order-1 in orders: pairs.append((i-1,i))

PSP,PCOV = get_P_spacings_with_covariance(F, FCOV, pairs)

L = len(PSP)
X = np.array([1e-4/sqrt(L) for x in PSP])
Z = np.zeros((L,L))
for i in range(L): Z[i,i] = PCOV[i,i]

S = np.linalg.inv(PCOV)
d1 = sqrt((S.dot(X)).dot(X))
print('%.2e'%d1)
ZINV = np.linalg.inv(Z)
d2 = sqrt((ZINV.dot(X)).dot(X))
print('%.2e'%d2)
print(d1/d2)
exit()

print(day*PSP)

im=plt.imshow(PCOV, interpolation=None, origin='lower')
plt.colorbar(im)
plt.title('Period spacing covariance')
plt.show()












