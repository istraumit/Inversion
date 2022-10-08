import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from bisect import bisect
from numpy.polynomial.chebyshev import chebval
from test_Omega import test_Omega
from scipy.optimize import curve_fit
import sobol_seq


if False:
    for i in range(10):
        Cheb_coefs = np.random.rand(5) - 0.5
        Cheb_x = np.linspace(-1, 1, 100)
        Cheb_poly = chebval(Cheb_x, Cheb_coefs)

        plt.plot(Cheb_x, Cheb_poly, color='blue')

    plt.show()

def fit_func(dummy_variable, *coefs):
    xx = np.linspace(-1, 1, 100)
    Cheb_poly = chebval(xx, coefs)
    return Cheb_poly

N_Cheb_coef = 5
xx = np.linspace(0, 1, 100)
sigma = np.array([1.0 for x in range(100)])
ydata = 0.01*test_Omega(xx)

if bool(1):
    che_coef = []
    che_coef.append((-275.872046527109, 2181.317515779611, 1585.8001052080554))
    che_coef.append((5.567648132260473, 32.55042892638374, 27.191041976021488))
    che_coef.append((-0.006715315600022528, 0.045390506206946665, 0.05392369012067444))
    che_coef.append((2.2063916645356806e-06, 3.201170380043262e-05, 2.6897181530062565e-05))
    che_coef.append((-1.6404307737527511e-10, 6.753034258975771e-09, 8.081934683295294e-09))
    che_coef.append((-1.3340532129751743e-14, 7.397081924350028e-13, 6.168739992764606e-13))
    che_x = np.linspace(-1, 1, 100)
    poly = chebval(che_x, [t[0] for t in che_coef])
    plt.plot(che_x, poly)
    plt.show()
    exit()

if bool(1):
    p0 = np.array([0.0 for x in range(N_Cheb_coef)])
    tol = 5.e-4

    popt, pcov = curve_fit(fit_func, xdata=[], ydata = ydata, sigma = sigma, p0 = p0, ftol = tol, xtol = tol, absolute_sigma = True, method = 'trf')

    print(popt)
    yy = fit_func(0, *popt)

    plt.plot(xx, ydata)
    plt.plot(xx, yy)
    plt.show()
    exit()

sobol_grid = 2*sobol_seq.i4_sobol_generate(N_Cheb_coef, 1000000) - 1

best_chi2 = 1e10

for i in range(sobol_grid.shape[0]):
    #point = 2*np.random.rand(N_Cheb_coef) - 1 
    point = sobol_grid[i,:]
    model = fit_func(0, *point)
    chi2 = np.sum((model-ydata)**2)
    if chi2 < best_chi2:
        best_chi2 = chi2
        best_model = point

print('Best chi2:', best_chi2)
yy = fit_func(0, *best_model)

plt.plot(xx, ydata)
plt.plot(xx, yy)
plt.show()






















