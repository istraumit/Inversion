import numpy as np
import sympy as sym
from sympy.vector import CoordSys3D, Del
import matplotlib.pyplot as plt
from magnetic_diff_xi import diff_xi, kernels

csys = CoordSys3D('csys')

B = csys.k # toroidal magnetic field configuration

r = sym.symbols('r')
theta = sym.symbols('theta')
phi = sym.symbols('phi')

Y11 = sym.S(0.3454941494713355) * sym.sqrt(1 - sym.cos(theta)**2) * sym.cos(phi)

dY_dtheta = sym.diff(Y11, theta)
dY_dphi = sym.diff(Y11, phi)

xi_r = sym.symbols('d_rxi_r_dr(r)')
xi_h = sym.symbols('xi_h(r)')
xi = xi_r * csys.i + xi_h * sym.sqrt(4*sym.pi) * (dY_dtheta * csys.j + dY_dphi * csys.k/sym.sin(theta))

xi_cross_B = xi.cross(B)
#print(xi_cross_B)
#print('-'*25)
#print(xi_cross_B.subs({theta:0.1, phi:0.1, xi_r:0}).evalf())

nabla_A_1 = sym.diff(xi_cross_B.components[csys.i], phi) / (r * sym.sin(theta))
nabla_A_2 = sym.diff(xi_cross_B.components[csys.i], theta) / r

I = nabla_A_1**2 + (1/r)*(xi_r - nabla_A_2)**2
print(I)
exit()

order = -20
mode = kernels[order]
rr = mode.r_coord
r_xi = mode.r_coord * mode.xi_r
drxirdr = diff_xi(rr, r_xi, rr)

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
            sub = {r:vr, theta:vtheta, phi:vphi, xi_h:mode.xi_h[ir], xi_r:drxirdr[ir]}
            Iv = I.subs(sub)
            M[iphi,ir] = Iv.evalf()

plt.imshow(M)
plt.show()
















