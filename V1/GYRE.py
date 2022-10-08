import os, sys
from math import *
import numpy as np
import matplotlib.pyplot as plt


def nHz_to_rads(omega):
    return 2 * pi * 1.e-9 * omega

def step_profile(r_norm):
    rr = [0.0, 0.1345, 1.0]
    prof = [1000., 250., 250.]
    assert len(prof)==len(rr)

    prof_rads = [nHz_to_rads(om) for om in prof]

    for i in range(len(rr)-1):
        if rr[i] <= r_norm <= rr[i+1]: return prof_rads[i]

def sin_profile(rr):
    return nHz_to_rads( 1000.0 + 1000.0 * np.sin(2 * pi * rr) )

def const_profile(r_norm):
    return nHz_to_rads(1000)

def linear_profile(r):
    return nHz_to_rads(1000. * r)

def add_rotation_to_pulse_file(path, postfix, rot_profile):
    path_out = path + '.omega.' + postfix
    with open(path) as f_in:
        with open(path_out, 'w') as f_out:
            n=0
            for line in f_in:
                n+=1
                if n==1:
                    f_out.write(line)
                    R_star = float(line.split()[2])
                else:
                    row = line.split()
                    r = float(row[1])
                    r_norm = r/R_star
                    omega = rot_profile(r_norm)
                    row[-1] = '%.16e'%omega

                    rowj = []
                    rowj.append(row[0].rjust(6))
                    rowj.extend([s.rjust(5+2+16+4) for s in row[1:]])

                    line_out = ''.join(rowj)
                    f_out.write(line_out)
                    f_out.write('\n')

if __name__=='__main__':
    D = {}
    D['const'] = const_profile
    D['step1'] = step_profile
    D['sin'] = sin_profile
    D['lin'] = linear_profile

    path = sys.argv[1]
    postfix = sys.argv[2]

    add_rotation_to_pulse_file(path, postfix, D[postfix])













