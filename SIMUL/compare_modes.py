import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d


class Eigenmode:

    def __init__(self, fn):
        with open(fn) as f:
            n = 0
            for line in f:
                n += 1
                if n==3: header = line.split()
                if n==4:
                    values = line.split()
                    break
        self.l = int(values[header.index('l')])
        self.m = int(values[header.index('m')])
        self.n_pg = int(values[header.index('n_pg')])

        data = np.loadtxt(fn, skiprows=6)
        xiR, xiH, rho, r = data[:,0], data[:,2], data[:,4], data[:,5]
        r_new = np.linspace(min(r), max(r), 10000)
        xiR = self.interpolate(r, xiR, r_new)
        xiH = self.interpolate(r, xiH, r_new)
        rho = self.interpolate(r, rho, r_new)
        r = r_new

        L2 = self.l*(self.l + 1)
        R = (xiR**2 + L2 * xiH**2 - 2*xiR*xiH - xiH**2) * r**2 * rho
        R_int = simps(R, r)
        self.kernel = R / R_int

        Q = (xiR**2 + L2 * xiH**2) * r**2 * rho
        Q_int = simps(Q, r)
        self.beta = R_int / Q_int
        self.r_coord = r
        self.xi_r = xiR
        self.xi_h = xiH
        self.rho = rho

    def interpolate(self, xx, ff, xx_new):
        interp = interp1d(xx, ff, kind='cubic')
        kern = interp(xx_new)
        return kern


def find_file(path, nlm):
    ff = [fn for fn in os.listdir(path) if nlm in fn]
    if len(ff)>1: print(path, ff)
    if len(ff)==0:
        print('Not found:', nlm)
    return ff[0]


#dpath1 = '/home/elwood/Documents/Inversion/DATA/03_GYRE_modes/M1.5_XC0.2'
dpath2 = '/home/elwood/Documents/Inversion/DATA/03_GYRE_modes/M3.0_XC0.6'
dpath1 = dpath2

nlm1 = '-00010_002_+00_00024'
nlm2 = '-00010_002_+00_00058'

path1 = os.path.join(dpath1, find_file(dpath1, nlm1))
path2 = os.path.join(dpath2, find_file(dpath2, nlm2))

m0 = Eigenmode(path1)
m1 = Eigenmode(path2)

plt.subplot(311)
plt.plot(m0.r_coord, m0.xi_r, label=nlm1)
plt.plot(m1.r_coord, m1.xi_r, label=nlm2)
#plt.xlabel('r')
plt.ylabel('xi_r')
plt.legend()

plt.subplot(312)
plt.plot(m0.r_coord, m0.xi_h, label=nlm1)
plt.plot(m1.r_coord, m1.xi_h, label=nlm2)
#plt.xlabel('r')
plt.ylabel('xi_h')
#plt.legend()

plt.subplot(313)
plt.plot(m0.r_coord, m0.kernel)
plt.plot(m1.r_coord, m1.kernel)
plt.xlabel('r')
plt.ylabel('Kernel')

plt.show()










