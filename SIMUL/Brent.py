import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def find_roots(F, xx):
    yy = [F(x) for x in xx]
    roots = [brentq(F, xx[i-1], xx[i]) for i in range(1, len(xx)) if yy[i-1]*yy[i] < 0]
    return roots


def func(x):
    return 1./(1. - x)


xx = np.linspace(0, 2, 100)
yy = func(xx)
plt.plot(xx, yy)

roots = find_roots(func, xx)
[print(r) for r in roots]
print('-'*25)
print(len(roots), 'roots found')

for r in roots: plt.axvline(r, color='red')
plt.axhline(0.0)
plt.show()
