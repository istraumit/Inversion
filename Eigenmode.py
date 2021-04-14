import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

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


if __name__=='__main__':

    E = Eigenmode('/home/elwood/Documents/Inversion/DATA/02_GYRE_omega_zero/pulse_M3.1874_Xc0.620927/mode_-00032_001_+00')
    print(E.l, E.m, E.n_pg, E.beta)
    plt.plot(E.r_coord, E.kernel)
    plt.show()



