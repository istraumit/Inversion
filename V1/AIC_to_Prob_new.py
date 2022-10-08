import sys
import math
import numpy as np

LN_P_ERR = -20.0
latex = True

def AIC(chi2, K, n, err):
    return chi2 + 2*K + ( 2*K*(K+1) )/(n - K - 1) + err

class Record:
    def __str__(self):
        if latex:
            return '%s & %i & %.1f & %.2f \\\\'%(self.rem, self.K, self.chi2, 100*self.P)
        else:
            return '%s \t %i \t %.1f \t %.1f \t %.2f'%(self.rem, self.K, self.chi2, self.aic, 100*self.P)

    def __repr__(self):
        return self.__str__()

def load_LOG(fn):
    data = {}
    with open(fn) as f:
        for line in f:
            if line.startswith('N ='):
                K = int(line.split()[2])
            if line.startswith('remove'):
                rem = line.split()[2]
            if line.startswith('CHI^2'):
                if not rem in data: data[rem] = []
                CHI = float(line.split()[2])
                rec = Record()
                rec.rem = rem
                rec.K = K
                rec.chi2 = CHI
                data[rem].append( rec )
    return data


param_chi2 = load_LOG(sys.argv[1])

n = int(sys.argv[2])

table = []
for p in param_chi2:
    i_min = np.argmin([t.chi2 for t in param_chi2[p]])
    t = param_chi2[p][i_min]
    if latex:
        print(p, '&', '%.0f'%t.chi2, '\\\\')
    else:
        print(p, '\t', t.K, '\t', t.chi2)
    table.extend(param_chi2[p])


for t in table:
    n_eff = n - 1
    err = -LN_P_ERR
    if t.rem == 'none':
        n_eff = n
        err = 0
    aic = AIC(t.chi2, t.K, n_eff, err)
    t.aic = aic

aic_min = min([t.aic for t in table])

for t in table:
    t.P = math.exp(-0.5*(t.aic - aic_min))

P_sum = sum([t.P for t in table])

print('-'*25)
print('P_sum = %.2e'%P_sum)
print('-'*25)

for t in table:
    t.P /= P_sum

table.sort(key = lambda t:-t.P)


for t in table:
    if 100*t.P > 0.1:
        print(t)



