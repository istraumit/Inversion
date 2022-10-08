import re
from math import *
import phys_const_cgs as co
import numpy as np
import matplotlib.pyplot as plt

with open('pulse.mesa') as f:
    s = f.readline()
    arr = s.split()
    M = float(arr[1]) # g
    R = float(arr[2]) # cm
    L = float(arr[3]) # erg/s

print('-'*16)
print('M =', '%.2f'%(M/co.Msol), 'M_sol')
print('R =', '%.2f'%(R/co.Rsol), 'R_sol')

rho = M/((4/3) * pi * R**3)

print('rho =', '%.2f'%(rho), 'g/cm^3')

# G [cm^3/g/s^2]
# G*rho [1/s^2]

f0 = sqrt(co.G * rho)
P = 1/f0
print('-'*16)
print('f =', 1e6*f0, 'uHz')
print('P =', P, 's')
print('-'*16)

Q = 1e6 * f0 / pi

freqs = []
regex_freq = re.compile('^.+?(\d+)\s+(\d\.\d+E[+-]\d{3}).+')
with open('summary.txt') as f:
    for line in f:
        m = regex_freq.match(line)
        if m:
            freqs.append( (int(m.groups()[0]), Q*float(m.groups()[1])) )
freqs.sort()
freqs.reverse()
print('n_g \tFreq [uHz]')
for freq in freqs:
    n_g = freq[0]
    f32 = freq[1]
    print( '%i\t%.3f'%(n_g, f32) )

freqs_filter = [x[1] for x in freqs if 14<=x[0]<=32]
triana = np.loadtxt('/STER/ilyas/asteroseismology/Triana/010526294/_triana.freq')
triana = triana[:,0]
ff = np.array(freqs_filter)

D = np.sum( abs(ff-triana) )
print('-'*16)
print('D =', D)


plt.subplot(121)
plt.plot(ff, triana, 'o')
plt.plot([ff[0], ff[-1]], [ff[0], ff[-1]])
plt.xlabel('Frequency [uHz]')
plt.ylabel('Frequency [uHz]')

plt.subplot(122)
plt.plot(ff, ff-triana, 'o')
plt.axhline(0)
plt.xlabel('Frequency [uHz]')
plt.ylabel('Frequency difference [uHz]')

plt.show()










