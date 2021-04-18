import os,sys
import numpy as np
import matplotlib.pyplot as plt
from SVD import truncated_SVD

N = 100

if os.path.isfile('K.npy'):
    Kmax = np.load('K.npy')
else:
    kappa_max = 0
    Kmax = None
    for i in range(100):
        K = np.random.rand(N,N)
        kappa = np.linalg.cond(K)
        if kappa > kappa_max:
            kappa_max = kappa
            Kmax = K
    np.save('K', Kmax)

print('kappa = %.2e'%np.linalg.cond(Kmax))

xx = np.linspace(-np.pi, np.pi, N)
X = 100*np.sin(xx)

J = np.dot(Kmax, X) + np.random.randn(N)

Kinv = np.linalg.inv(Kmax)
Xrec = np.dot(Kinv, J)

Xlstsq,res,rank,sv = np.linalg.lstsq(Kmax, J, rcond=0.01)
print('Rank =', rank)

plt.plot(X, 'o-', label='Original')
plt.plot(Xrec, 'o-', label='Inverse')
plt.plot(Xlstsq, 'o-', label='Least sq')
plt.legend()
plt.show()







