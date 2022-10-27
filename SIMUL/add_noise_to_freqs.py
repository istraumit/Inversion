import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon


def exp_distr(xx, lam):
    return lam * np.exp(-lam*xx)

def add_noise(path, noise):
    path_out = path + '.noise'
    tmpl = '0.0000000000000000E+000'
    remove_end = 2*(len(tmpl) + 3)
    with open(path) as f_in:
        with open(path_out, 'w') as f_out:
            n=0
            for line in f_in:
                n+=1
                if n<7:
                    f_out.write(line)
                else:
                    arr = line.split()
                    freq = float(arr[7])
                    freq_noisy = freq + np.random.randn()*noise[n]
                    new_line = line.rstrip()[:-remove_end] + '   ' + '%.16e'%freq_noisy + '   '+tmpl+'\n'
                    f_out.write(new_line)

if __name__=='__main__':

    day = 24 * 60 * 60

    def cd_to_Hz(cd):
        return cd / day

    err = cd_to_Hz(np.loadtxt('Kurtz_2014_g_mode_errors_cyc_day'))
    mean = np.mean(err)
    lam = 1./mean

    if False:
        xx = np.linspace(0, max(err), 100)
        yy = exp_distr(xx, lam)

        plt.hist(err, density=True, label='Kurtz et al 2014 data', alpha=0.5)
        plt.plot(xx, yy, linewidth=2, label='Exp distribution')
        plt.xlabel('Error [cyc/day]')
        plt.legend()
        plt.show()

        sample = expon.rvs(size=1000, scale=mean)
        plt.hist(sample, 50, density=True, alpha=0.5, label='Random sample')
        plt.plot(xx, yy, linewidth=2, label='Exp distribution')
        plt.legend()
        plt.xlabel('Error [cyc/day]')
        plt.xlim(0, xx[-1])
        plt.show()

    path = sys.argv[1]
    add_noise(path, expon.rvs(size=1000, scale=mean))



