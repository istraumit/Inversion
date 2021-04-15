import numpy as np

def get_P_spacings(freq, freq_pairs):
    P_spacings = []
    for pair in freq_pairs:
        P0 = 1./freq[pair[0]]
        P1 = 1./freq[pair[1]]
        P_sp = P0 - P1
        P_spacings.append(P_sp)
    return np.array(P_spacings)

def get_P_spacings_with_covariance(frequencies, freq_covariance_matrix, freq_pairs, N_MC_samples=10000):
    """
    frequencies: 1D array with frequencies
    freq_covariance_matrix: 2D array with the frequency covariance structure
    freq_pairs: list of tuples (i,j) where i,j are indices in the frequency array, 
        specifying between which pairs of frequencies to calculate the period spacings
    N_MC_samples: number of samples, the more the better
    """
    assert len(frequencies.shape)==1
    assert len(freq_covariance_matrix.shape)==2
    assert frequencies.shape[0]==freq_covariance_matrix.shape[0]==freq_covariance_matrix.shape[1]
    assert type(freq_pairs)==list
    assert all([type(p)==tuple and len(p)==2 for p in freq_pairs])

    P_sp_sample = []
    for i in range(N_MC_samples):
        F = np.random.multivariate_normal(frequencies, freq_covariance_matrix)
        P_sp = get_P_spacings(F, freq_pairs)
        P_sp_sample.append(P_sp)

    P_sp_sample = np.array(P_sp_sample)
    P_sp_exact = get_P_spacings(frequencies, freq_pairs)
    P_sp_cov = np.cov(P_sp_sample, rowvar=False)

    return P_sp_exact, P_sp_cov


