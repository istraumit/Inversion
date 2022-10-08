import os
import numpy as np
from inversion import ModeKernel
import matplotlib.pyplot as plt
from bisect import bisect

modes_dir = '../gyre_work/KIC105/Moravveji_best_model/modes'
modes_dir = '../gyre_work/KIC105_lowXc/pulse_M3.2365_Xc0.241902'
print(modes_dir)

MK = ModeKernel()
kernels = {}
mode_files = os.listdir(modes_dir)
for fn in mode_files:
    if not fn.startswith('mode'): continue
    if not os.path.isfile(os.path.join(modes_dir, fn)): continue
    kern_data = MK.get_kernel_and_beta(os.path.join(modes_dir, fn))
    order = kern_data.n_pg
    kernels[order] = kern_data

def diff_xi(xi_r, xi, rr):
    deg = 2
    res = []
    for r in rr:
        i = bisect(xi_r, r)
        if i<2: i=2
        if i>len(xi)-3: i = len(xi)-3
        xi_slice = xi[i-2:i+3]
        xi_r_slice = xi_r[i-2:i+3]
        pfit = np.polynomial.polynomial.Polynomial.fit(xi_r_slice, xi_slice, deg)
        coef = pfit.convert().coef
        deriv = np.sum([coef[n] * n * r**(n-1) for n in range(1, deg+1)])
        res.append(deriv)
    return np.array(res)

if __name__=='__main__':

    rr = kernels[-20].r_coord # np.linspace(0, 1, 1000)
    r_xi = kernels[-20].r_coord * kernels[-20].xi_r

    deriv = diff_xi(rr, r_xi, rr)

    plt.plot(rr, r_xi, '.-')
    plt.plot(rr, 0.05*deriv, '.-')
    plt.axhline(0, color='black')

    plt.show()







