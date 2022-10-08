import sys
import math

def AIC(chi2, K, n):
    return chi2 + 2*K + ( 2*K*(K+1) )/(n - K - 1)

def load_LOG(fn):
    data = []
    with open(fn) as f:
        for line in f:
            if line.startswith('N ='):
                N = int(line.split()[2])
            if line.startswith('CHI^2'):
                CHI = float(line.split()[2])
                data.append( (N, CHI) )
    return data


#---------------- Triana
param_chi2 = []
param_chi2.append( (2, 8.1) )
param_chi2.append( (3, 4.6) )
param_chi2.append( (4, 5.8) )
param_chi2.append( (5, 4.0) )
param_chi2.append( (6, 5.3) )
param_chi2.append( (7, 4.2) )
param_chi2.append( (8, 4.0) )
param_chi2.append( (9, 2.5) )
param_chi2.append( (10, 4.6) )
param_chi2.append( (11, 4.3) )

#-----------------Kurtz
param_chi2 = []
param_chi2.append( (2, 2953) )
param_chi2.append( (3, 1303) )
param_chi2.append( (4, 395) )
param_chi2.append( (5, 390) )
param_chi2.append( (6, 89) )
param_chi2.append( (7, 116) )
param_chi2.append( (8, 61) )
param_chi2.append( (9, 71) )
param_chi2.append( (10, 10.5) )
param_chi2.append( (11, 14) )
param_chi2.append( (12, 24.8) )
param_chi2.append( (13, 186) )
param_chi2.append( (14, 117) )

param_chi2 = load_LOG(sys.argv[1])

n = int(sys.argv[2])

aics = []
for i,v in enumerate(param_chi2):
    aic = AIC(v[1], v[0], n)
    aics.append(aic)

aic_min = min(aics)

delta = [aic - aic_min for aic in aics]

LL = [math.exp(-0.5*d) for d in delta]

LL_sum = sum(LL)
P = [L/LL_sum for L in LL]

for i,v in enumerate(param_chi2):
    print( '%i & %.1f & %.0f & %.2f \\\\'%(v[0], v[1], aics[i], 100*P[i]) )


