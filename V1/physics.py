import math
import numpy as np
import matplotlib.pyplot as plt
from phys_const import Msol, Rsol, Lsol, G, sigma

M = 3.2 * Msol
R = 2.1 * Rsol
T = 12470
L = 4 * math.pi * R**2 * sigma * T**4
t_KH = G * M**2 / R / L
year = 3600 * 24 * 356.25
print('t_KH =', '%.2e'%(t_KH/year), 'years')
Om = 60.e-9
t_ES = t_KH * G * M / Om**2 / R**3
print('t_ES =', '%.2e'%(t_ES/year), 'years')


Om_c = math.sqrt(2**3 * G * M / 3**3 / R**3) / (2 * math.pi)
print('Omega_c =', Om_c * 1.e9, 'nHz')

r_bounds = [0.159, 0.264125, 0.36924999999999997, 0.474375, 0.5795, 0.684625, 0.78975, 0.894875, 1.0]
r_points = []
for i in range(1, len(r_bounds)):
    r_points.append( 0.5*(r_bounds[i] + r_bounds[i-1]) )

Omega = []
Omega.append((162.00178222164806, 591.5854557650496, 594.1989705805181))
Omega.append((410.5517298116768, 338.0638815053889, 338.36019864083835))
Omega.append((388.18268780682786, 427.0628582347514, 426.7448262299407))
Omega.append((94.92670541916073, 569.0362095523037, 569.3444318616082))
Omega.append((-206.22902577280655, 645.4195657725212, 642.0633832902986))
Omega.append((-370.49779760000786, 488.27752351480746, 482.79394371563194))
Omega.append((-574.2120655654597, 692.0709390262599, 678.2959565794268))
Omega.append((-1029.395053414932, 687.6514019381003, 689.7508111076534))

poly = np.polyfit(r_points, [O[0] for O in Omega], deg=3, w=[1.0/O[1] for O in Omega])
print(poly)
xx_poly = np.linspace(min(r_bounds), max(r_bounds), 1000)
yy_poly = np.poly1d(poly)(xx_poly)

plt.errorbar(r_points, [O[0] for O in Omega], yerr=[O[1] for O in Omega], fmt='o')
plt.plot(xx_poly, yy_poly, linewidth=2)
plt.xlabel('Radius [R_star]')
plt.ylabel('Omega [nHz]')
plt.show()

s = xx_poly
Om = yy_poly


N_om_2 = np.gradient( (s**2 * Om)**2 ) / (s**3)

plt.plot(s, N_om_2)
plt.axhline(0.0, color='black')
plt.xlabel('Radius [R_star]')
plt.ylabel('N_Omega^2')
plt.grid()
plt.show()






data = np.loadtxt('pulse.mesa', skiprows=1)

r = data[:,1]
r /= max(r)
rho = data[:,6]
N2 = 1.e18*data[:,8]

N_Om_2 = np.interp(r, s, N_om_2)

plt.plot(r, N2 + N_Om_2)
plt.grid()
plt.show()


















