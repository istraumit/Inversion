import os
import math
import matplotlib.pyplot as plt
#from Jordan import get_Jordan_triplets

day = 24 * 60 * 60

def nHz_to_cd(f):
    return day*1.e-9*f

def cd_to_Hz(cd):
    return cd / day

def P_to_nHz(P):
    return 1.e9*cd_to_Hz(1./P)

def cd_to_nHz(f):
    return 1.e9*cd_to_Hz(f)

def P_to_nHz_tuple(t):
    return (P_to_nHz(t[0]), 1.e9*cd_to_Hz( t[1] / t[0]**2) )

class SplitData:
    def __init__(self):
        self.center_freq = -1
        self.left_split = -1
        self.right_split = -1

        self.center_error = -1
        self.left_error = -1
        self.right_error = -1

        self.error_inflate_factor = 30.0

    def __str__(self):
        L = self.left_split_error()
        R = self.right_split_error()
        return '%.6f +/- %.6f: %.6f +/- %.6f, %.6f +/- %.6f'%(self.center_freq, self.center_error, self.left_split, L, self.right_split, R)

    def __repr__(self):
        return self.__str__()

    def check(self):
        if self.left_split<0 and self.right_split>0:
            self.left_split = self.right_split
        if self.left_split>0 and self.right_split<0:
            self.right_split = self.left_split

        if self.left_error<0 and self.right_error>0:
            self.left_error = self.right_error
        if self.left_error>0 and self.right_error<0:
            self.right_error = self.left_error
        if self.center_error < 0:
            self.center_error = 0.000019

    def left_split_error(self):
        return self.error_inflate_factor * math.sqrt(self.center_error**2 + self.left_error**2)

    def right_split_error(self):
        return self.error_inflate_factor * math.sqrt(self.center_error**2 + self.right_error**2)


    def to_Hz(self):
        s = SplitData()
        s.center_freq = cd_to_Hz(self.center_freq) * 1.e6 # microHz
        s.center_error =  cd_to_Hz(self.center_error) * 1.e6 # microHz

        s.left_split =  cd_to_Hz(self.left_split) * 1.e9 # nHz
        s.right_split =  cd_to_Hz(self.right_split) * 1.e9 # nHz
        s.left_error =  cd_to_Hz(self.left_error) * 1.e9 # nHz
        s.right_error =  cd_to_Hz(self.right_error) * 1.e9 # nHz
        return s

def load_papics(to_Hz=True):
    digits = [str(i) for i in range(10)]
    split = {}
    enc = 'windows-1253'
    errors = {}
    with open('papics.errors', encoding=enc) as f:
        for line in f:
            arr = line.split()
            errors[arr[0]] = float(arr[1])

    with open('splittings.papics', encoding=enc) as f:
        for line in f:
            arr = line.split()
            txt = arr[0][1:]
            n_txt = txt[0]
            if len(txt)>1 and txt[1] in digits:
                n_txt += txt[1]
            n = -(33 - int(n_txt))
            PM = txt[-1]
            if PM in digits: PM = 'c'

            err = -1
            if len(arr) > 1:
                if arr[1] in errors:
                    err = errors[arr[1]]
                else:
                    #print('error missing for ', arr[1])
                    pass

            if not n in split: split[n] = SplitData()
            if PM == 'c':
                split[n].center_freq = float(arr[1])
                split[n].center_error = err
            if PM == '-' and len(arr)>3:
                split[n].left_split = float(arr[3])
                split[n].left_error = err
            if PM == '+' and len(arr)>3:
                split[n].right_split = float(arr[3])
                split[n].right_error = err

    for k in split:
        split[k].check()
        if to_Hz:
            split[k] = split[k].to_Hz()
    return split


def get_diff_error(e1, e2):
    """
    Returns error of a difference of two Gaussian random variables
    ----------------
    e1, e2: errors of the input variables
    """
    return 3.0 * math.sqrt(e1**2 + e2**2)


def load_kurtz_g():
    freq = {}
    with open('splittings.kurtz') as f:
        for line in f:
            arr = line.split()
            if len(arr)==0: break
            n = (int(arr[0]), int(arr[1]))
            if not n in freq: freq[n] = []
            freq[n].append( (1.e9*cd_to_Hz(float(arr[3])), 1.e9*cd_to_Hz(float(arr[4]))) )

    split = {}
    for n in freq:
        assert len(freq[n])==2
        sp = 0.5*abs(freq[n][0][0] - freq[n][1][0])
        er = get_diff_error(freq[n][0][1], freq[n][1][1])
        split[n] = (sp, er)

    assert len(split)==15

    return split

# product of two Gaussian distributions
#   mu1, sig1 - mean and std.dev. of the distribution #1
#   mu2, sig2 - mean and std.dev. of the distribution #2
def G_product(mu1, mu2, sig1, sig2):
   sig = 1/(1/sig1**2 + 1/sig2**2)
   mu = sig * (mu1/sig1**2 + mu2/sig2**2)
   return mu, math.sqrt(sig)

# merge a set of values with errors
#   mu - list with the values
#   sig - list with the errors
def merge_errors(mu, sig):
   m = mu[0]
   s = sig[0]
   for i in range(1, len(mu)):
      m, s = G_product(m, mu[i], s, sig[i])
   return (m, s)


def load_kurtz_p_l2():
    freq = {}
    with open('splittings.kurtz') as f:
        for line in f:
            arr = line.split()
            if len(arr)==0: continue
            l = int(arr[0])
            n = int(arr[1])
            nl = (l, n)
            if n < -15: continue
            if l != 6:
                if not nl in freq: freq[nl] = []
                freq[nl].append( (1.e9*cd_to_Hz(float(arr[3])), 1.e9*cd_to_Hz(float(arr[4]))) )

    split = {}
    for n in freq:
        if len(freq[n])==5:
            sp, er = [],[]
            sp.append(freq[n][2][0] - freq[n][1][0])
            sp.append(freq[n][3][0] - freq[n][2][0])
            sp.append(0.5*(freq[n][2][0] - freq[n][0][0]))
            sp.append(0.5*(freq[n][4][0] - freq[n][2][0]))

            er.append( get_diff_error(freq[n][2][1], freq[n][1][1]))
            er.append( get_diff_error(freq[n][3][1], freq[n][2][1]))
            er.append( 0.5*get_diff_error(freq[n][2][1], freq[n][0][1]))
            er.append( 0.5*get_diff_error(freq[n][4][1], freq[n][2][1]))

            split[n] = merge_errors(sp, er)
        elif len(freq[n])==3:
            sp,er = [],[]
            sp.append(freq[n][1][0] - freq[n][0][0])
            sp.append(freq[n][2][0] - freq[n][1][0])

            er.append(get_diff_error(freq[n][1][1], freq[n][0][1]))
            er.append(get_diff_error(freq[n][2][1], freq[n][1][1]))

            split[n] = merge_errors(sp, er)
        else:
            print('not 3 or 5:'+str(n)+' '+str(freq[n]))

    return split



def load_split_KIC9751996():
    fn = os.path.join('KIC_9751996','mode_ident','Kepler9751996_split.dat')
    m_plus = {}
    m_zero = {}
    m_minus = {}
    with open(fn) as f:
        for line in f:
            if line.startswith('-'):
                arr = line.split()
                n = int(arr[0])
                m = int(arr[1])
                P = float(arr[2])
                P_err = float(arr[3])
                if m == -1:
                    m_minus[n] = P_to_nHz_tuple((P, P_err))
                if m == 0:
                    m_zero[n] = P_to_nHz_tuple((P, P_err))
                if m == 1:
                    m_plus[n] = P_to_nHz_tuple((P, P_err))

    plus_set = set(m_plus.keys())
    minus_set = set(m_minus.keys())
    common_set = plus_set & minus_set
    common_list = list(common_set)
    common_list.sort()
    split = {}
    for n in common_list:
        if n in m_zero:
            sp_left = m_zero[n][0] - m_minus[n][0]
            sp_left_err = get_diff_error(m_zero[n][1], m_minus[n][1])

            sp_right = m_plus[n][0] - m_zero[n][0]
            sp_right_err = get_diff_error(m_zero[n][1], m_plus[n][1])

            sp, sp_err = G_product(sp_left, sp_right, sp_left_err, sp_right_err)
        else:
            sp = 0.5*(m_plus[n][0] - m_minus[n][0])
            sp_err = 0.5*get_diff_error(m_plus[n][1], m_minus[n][1])

        split[n] = (sp, sp_err)

    return split


def load_split_KIC10080943(component):
    fn = 'KIC_10080943/splittings_'+component
    split = {}
    with open(fn) as f:
        for line in f:
            arr = line.split()
            n = -1*int(arr[0])
            val = float(arr[-3])
            err = float(arr[-1])
            split[n] = (cd_to_nHz(val), cd_to_nHz(err))
    return split

def load_Jordan_split(kind, inflation_factor=1.0):
    trip = get_Jordan_triplets()
    split = {}
    for order in trip:
        t = trip[order].jordan_triplet
        if t==None: continue
        #print(t.cov)
        if kind=='left':
            mu = t.freq[1]-t.freq[0]
            var = t.cov[1,1] + t.cov[0,0] - 2*t.cov[0,1]
        elif kind=='right':
            mu = t.freq[2]-t.freq[1]
            var = t.cov[2,2] + t.cov[1,1] - 2*t.cov[1,2]
        elif kind=='symm':
            mu1 = t.freq[1]-t.freq[0]
            var1 = t.cov[1,1] + t.cov[0,0] - 2*t.cov[0,1]
            mu2 = t.freq[2]-t.freq[1]
            var2 = t.cov[2,2] + t.cov[1,1] - 2*t.cov[1,2]
            mu = 0.5*(mu1 + mu2)
            var = max(var1, var2)
        else:
            raise Exception('bad splitting kind ' + kind)
        split[-order] = (cd_to_nHz(mu), inflation_factor*cd_to_nHz(math.sqrt(var)))

    return split


def load_splittings_Triana():
    splittings = {}
    if True:
        with open('splittings.triana') as f:
            n = 0
            for line in f:
                n += 1
                if n < 3: continue
                arr = line.split()
                splittings[-int(arr[0])] = (float(arr[2]), float(arr[4]))
    return splittings



if __name__=='__main__':

    sp = load_splittings_Triana()

    print(sp)















