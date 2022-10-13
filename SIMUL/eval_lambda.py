import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess as sp



eval_lambda_path = '/home/elwood/Soft/gyre/6.0/bin/eval_lambda'
tmp_h5 = 'TMP_LAMBDA_EVAL.H5'
def call_eval_lambda(q_min, q_max, q_n, l, m):
    params = [l, m, q_min, q_max, q_n, 'T', 'F', tmp_h5]
    par = ' '.join([str(p) for p in params])
    o = sp.check_output(eval_lambda_path + ' ' + par, shell=True, stderr=sp.STDOUT)

    fh5 = h5py.File(tmp_h5, 'r')

    qq = list(fh5['q'])
    lam = list(fh5['lambda'])
    return (qq,lam)


class TidalLambda:
    
    def __init__(self):
        L = {}
        for l in [1,2]:
            L[l] = {}
            for m in [-1,0,1]:
                L[l][m] = call_eval_lambda(1.e-6, 100.0, 1000, l, m)
        self.L = L

    def get(self, qq, l, m):
        return np.interp(qq, self.L[l][m][0], self.L[l][m][1])














