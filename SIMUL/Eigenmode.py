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

if __name__=='__main__':
    from calculus import differentiate
    E = Eigenmode('/home/elwood/Documents/Inversion/gyre_work/rot_test/m=1/none/mode_-00032_001_+01')
    print(E.l, E.m, E.n_pg, E.beta)

    #diff = differentiate(E.r_coord, E.xi_h, E.r_coord)

    plt.plot(E.r_coord, E.kernel, '.-')
    #plt.plot(E.r_coord, diff)
    plt.show()









