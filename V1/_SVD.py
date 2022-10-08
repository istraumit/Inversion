import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def R_kernel(fn):
    with open(fn) as f:
        n = 0
        for line in f:
            n += 1
            if n==3: header = line.split()
            if n==4: 
                values = line.split()
                break
    l = int(values[header.index('l')])
    n_pg = int(values[header.index('n_pg')])

    data = np.loadtxt(fn, skiprows=6)
    xiR, xiH, rho, r = data[:,0], data[:,2], data[:,4], data[:,5]
    
    #plt.plot(r, xiH, label=str(n_pg))
    
    L2 = l*(l+1)
    R = (xiR**2 + L2 * xiH**2 - 2*xiR*xiH - xiH**2) * r**2 * rho
    R_int = simps(R, r)
    K = R/R_int
    
    Q = (xiR**2 + L2 * xiH**2) * r**2 * rho
    Q_int = simps(Q, r)
    beta = R_int / Q_int
    
    return r, K, n_pg, beta


folder = 'modes_no_rot'
files = [x for x in os.listdir(folder) if x.startswith('mode_')]
basis, betas = [],[]
for fn in files:
    r,R,n_pg,beta = R_kernel(os.path.join(folder, fn))
    betas.append((n_pg, beta))

    basis.append(R)
    #if n_pg in [-14]: plt.plot(r, R, label='n=%i'%n_pg, color='blue', alpha=1)
    #Q = np.vstack([r,R])
    #np.savetxt('ROT_'+fn, Q.T)
    #if n_pg<0: col = 'blue'
    #else: col='orange'
    #plt.plot(r, R, label=fn, color=col, alpha=0.25)

#plt.legend()
#plt.show()

G = np.array(basis)

svd = np.linalg.svd(G)
U = svd[0]
S = svd[1]
V = svd[2]

condi = max(S)/min(S)
print(condi)

plt.plot([i for i in range(1, 20)], S, 'o-')
plt.xlabel('Singular value index')
plt.ylabel('Value')
plt.show()
exit()



MP = np.linalg.pinv(G, rcond=0.001)
Rm = MP.dot(G)
Rd = G.dot(MP)

cov_m = MP.dot(MP.T)
m_sigma = np.sqrt(np.diagonal(cov_m))

target = 1 + 0.0*r
data = G.dot(target)
m = MP.dot(data)

plt.plot(m)
plt.show()
exit()

S /= max(S)

plt.plot(S, 'o-')
plt.axhline(0)
plt.show()






