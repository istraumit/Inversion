import numpy as np
import matplotlib.pyplot as plt

LIMIT = 0.00597 #c/d

class FreqItem:
    pass
    # triana_center_f: float
    # jordan_triplet: JordanTriplet

class JordanTriplet:
    pass
    # freq: 1D array
    # cov: 2D array

def find_triplet_jordan(cf, freq, cov):
    triplet_idx = []
    for k in range(3):
        dmin = np.inf
        imin = -1
        for i,v in enumerate(freq):
            if i in triplet_idx: continue
            #if i==34: continue
            d = abs(cf - v)
            if d < dmin:
                dmin = d
                imin = i
        if dmin > LIMIT: break
        triplet_idx.append(imin)

    if len(triplet_idx) < 3: return None

    triplet_idx = sorted(triplet_idx, key=lambda i:freq[i])

    jt = JordanTriplet()
    jt.freq = [freq[i] for i in triplet_idx]
    jt.cov = np.zeros((3,3))
    for i, iv in enumerate(triplet_idx):
        for j, jv in enumerate(triplet_idx):
            jt.cov[i,j] = cov[iv, jv]
    jt.idx = triplet_idx
    return jt

def get_Jordan_triplets():
    data = {}

    triana_center_f = np.loadtxt('Triana/triana.center.freq')
    for i in range(triana_center_f.shape[0]):
        order = int(triana_center_f[i,0])
        item = FreqItem()
        item.triana_center_f = triana_center_f[i,1]
        data[order] = item

    jordan = np.load('Triana/Jordan_covar_2.npz')
    jordan_cov = jordan['cov']
    jordan_freq = jordan['freq']

    for order in data:
        data[order].jordan_triplet = find_triplet_jordan(data[order].triana_center_f, jordan_freq, jordan_cov)

    return data


















