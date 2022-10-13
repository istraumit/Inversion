import numpy as np

def condi_num(K):
    svd = np.linalg.svd(K)
    S = svd[1]
    condi = max(S)/min(S)
    return condi

def truncated_SVD(K, n):
    U,S,V = np.linalg.svd(K)

    ST = np.zeros((U.shape[0], V.shape[0]))
    for i in range(n):
        ST[i,i] = 1./S[i]

    G = np.matmul(U, ST)
    Q = np.matmul(G, V)

    return Q.T # Omega = np.dot(Q.T, splittings)
