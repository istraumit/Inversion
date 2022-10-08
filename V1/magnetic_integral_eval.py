from math import *
import numpy as np
import matplotlib.pyplot as plt
from magnetic_diff_xi import diff_xi, kernels


order = -33
mode = kernels[order]
rr = mode.r_coord
r_xi = mode.r_coord * mode.xi_r
drxirdr = diff_xi(rr, r_xi, rr)

plt.plot(mode.r_coord, mode.xi_h)
plt.show()
exit()


def xi_h(ir):
    return mode.xi_h[ir]

def d_rxi_r_dr(ir):
    return drxirdr[ir]


def I(r, theta, phi, ir):
    return (d_rxi_r_dr(ir) - (-0.690988298942671*sqrt(pi)*xi_h(ir)*sin(theta)**2*cos(phi)/sqrt(1 - cos(theta)**2) 
    + 0.690988298942671*sqrt(pi)*xi_h(ir)*cos(phi)*cos(theta)**2/sqrt(1 - cos(theta)**2) 
    - 0.690988298942671*sqrt(pi)*xi_h(ir)*sin(theta)**2*cos(phi)*cos(theta)**2/(1 - cos(theta)**2)**(3/2))/r)**2/r 
    + 0.477464829275686*pi*xi_h(ir)**2*sin(phi)**2*cos(theta)**2/(r**2*(1 - cos(theta)**2))



N_thetas = 1000
N_phis = 1000
thetas = np.linspace(0.5*np.pi, np.pi, N_thetas)
phis = np.linspace(-np.pi, np.pi, N_phis)
dtheta = (max(thetas)-min(thetas))/N_thetas
dphi = (max(phis)-min(phis))/N_phis

M = np.zeros((N_phis, len(rr)))

for vtheta in [0.5*np.pi]:#thetas:
    for iphi,vphi in enumerate(phis):
        for ir, vr in enumerate(rr):
            if ir<10: continue
            M[iphi,ir] = log10( I(vr, vtheta, vphi, ir) )

im = plt.imshow(M.T, origin='lower', interpolation='none', extent=(min(phis), max(phis), min(rr), max(rr)), aspect='auto', cmap='jet')
bar = plt.colorbar(im, aspect=20)
bar.ax.set_ylabel('log(|B|^2)')

plt.xlabel('Phi [rad]')
plt.ylabel('r/R')
plt.show()




