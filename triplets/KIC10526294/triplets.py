import numpy as np
import matplotlib.pyplot as plt


LIMIT = 0.00597 #c/d

class Triplet:
    # freq: 1D array
    # cov: 2D array
    def __repr__(self):
        txt = []
        txt.append('Triplet n='+str(self.n))
        txt.append( str(self.freq) )
        txt.append( str(self.cov) )
        return '\n'.join(txt) + '\n'

def find_triplet(cf, freq, cov):
    triplet_idx = []
    for k in range(3):
        dmin = np.inf
        imin = -1
        for i,v in enumerate(freq):
            if i in triplet_idx: continue
            d = abs(cf - v)
            if d < dmin:
                dmin = d
                imin = i
        if dmin > LIMIT: break
        triplet_idx.append(imin)

    if len(triplet_idx) < 3: return None

    triplet_idx = sorted(triplet_idx, key=lambda i:freq[i])

    jt = Triplet()
    jt.freq = [freq[i] for i in triplet_idx]
    jt.cov = np.zeros((3,3))
    for i, iv in enumerate(triplet_idx):
        for j, jv in enumerate(triplet_idx):
            jt.cov[i,j] = cov[iv, jv]
    jt.idx = triplet_idx
    return jt

def load_triplets():
    center_freqs = {}
    triana_data = np.loadtxt('triana.center.freq')
    for i in range(triana_data.shape[0]):
        order = int(triana_data[i,0])
        center_freqs[order] = triana_data[i,1]

    jordan = np.load('Jordan_covar_2.npz')
    jordan_cov = jordan['cov']
    jordan_freq = jordan['freq']

    triplets = {}
    for order in center_freqs:
        t = find_triplet(center_freqs[order], jordan_freq, jordan_cov)
        if t!=None:
            t.n = order
            t.units = 'c/d'
            triplets[order] = t

    return triplets, jordan_freq, jordan_cov




















