import numpy as np

def load_pulse(path, ind):
    with open(path) as f:
        for line in f:
            header = line.split()
            break

    M = float(header[1])
    R = float(header[2])
    L = float(header[3])

    d = np.loadtxt(path, skiprows=1)
    rr = d[:,1]/R
    vv = d[:,ind]
    return rr, vv
