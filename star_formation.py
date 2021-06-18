import phys_const_cgs as co
from math import pi

M = 3.0 * co.Msol
R = 2.1 * co.Rsol

n_cloud = 1e4 # number density in the cloud

rho_cloud = n_cloud * co.mp

V_cloud = M/rho_cloud
R_cloud = (3*V_cloud/4/pi)**(1./3)

R_cloud_pc = R_cloud / co.pc

print('R cloud:', R_cloud_pc, '[pc]')

B_init = 1e-6 # initial B [G]

B = B_init * (R_cloud/R)**2

print('B:', '%.2e'%B, '[G]')


