import subprocess as sp
import numpy as np
import pickle

day = 24 * 60 * 60

def cd_to_Hz(cd):
    return cd / day

def cd_to_nHz(f):
    return 1.e9*cd_to_Hz(f)

def run(cmd):
    try:
        o = sp.check_output(cmd, shell=True)
    except sp.CalledProcessError as err:
        print(err.output)

def print_dict(d):
    for key in d:
        print(key, ':', d[key])

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def condi_num(K):
    svd = np.linalg.svd(K)
    S = svd[1]
    condi = max(S)/min(S)
    return condi
