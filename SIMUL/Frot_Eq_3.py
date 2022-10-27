import os
from Equation3 import *
from Figure import Figure
from scipy.integrate import simps
from calculus import integrate


data_dir = '/home/elwood/Documents/Inversion/DATA'
show_fig = True
alpha_g_plot = False
fmt = {1:'o-', -1:'s'}
uHz = 1.e6

def even(nn):
    return [n for n in nn if n%2==0]

def plot_freqs(S, nn, l, model):
    plt.plot(nn, [1.e9*(S[(l,1,n)]-S[(l,0,n)]) for n in nn], fmt[1], color='black', label='m=1')
    plt.plot(nn, [1.e9*(-S[(l,-1,n)]+S[(l,0,n)]) for n in nn], fmt[-1], color='black', label='m=-1')
    plt.legend()
    plt.xlabel('Radial order')
    plt.xticks(even(nn))
    plt.ylabel('Frequency splitting [nHz]')
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    plt.tight_layout()
    if save:
        plt.savefig('figures/l='+str(l)+'/'+model+'_split.pdf')
        plt.clf()
    else:
        plt.show()


def plot_alpha_g(model, lmn, ag_func):
    xx = np.linspace(0, 20./uHz, 50)
    yy = [ag_func(x) for x in xx]
    fig = Figure('f_rot [microHz]', 'F', title='(l,m,n) = '+str(lmn), legend=False)

    fig.add_series(xx * uHz, yy)

    fig.add_lines('v', [10.0], style='--', color='black')
    fig.add_lines('h', [0.0], style='-', color='black')

    if save:
        fig.dump('figures/roots/'+model+'_'+ str(lmn) +'.fig')
    else:
        fig.show()

class Model:
    def __init__(self, M, XC, rot):
        self.M = M
        self.XC = XC
        self.rot = rot
        self.noise = False

    def __str__(self):
        return 'M%sXC%sR%s'%(self.M, self.XC, self.rot)

    def label(self):
        return 'M=%s, Xc=%s'%(self.M, self.XC)

    def pulse_name(self):
        name = 'pulse_M' + self.M + '_XC' + self.XC + '.mesa'
        if self.rot!=None: name += '.omega.const.' + self.rot
        return name

    def freq_name(self):
        name = 'M' + self.M + '_XC' + self.XC
        if self.rot!=None: name += '.omega.const.' + self.rot
        name += '.out'
        if self.noise: name += '.noise'
        return name

    def get_rot_Hz(self):
        if self.rot==None: return 0.0
        return 1.e-6 * float(self.rot)

def prep_series(ser):
    n_max = max([len(x) for x in ser])
    for q in ser:
        L = len(q)
        fill = None
        if L>0: fill = q[-1]
        for i in range(n_max-L): q.append(fill)
    return ser

def norm_series(ser):
    mn = min(ser)
    mx = max(ser)
    return [ (x-mn)/(mx-mn) for x in ser]

#def norm_multi_series(ser):
#    a = np.array(ser)
    

def get_Pi0_precise(xx, NN, f_osc):
    yy = NN - f_osc
    def NN_f_osc(x):
        return np.interp(x, xx, yy)
    roots = [brentq(NN_f_osc, xx[i-1], xx[i]) for i in range(1, len(xx)) if yy[i-1]*yy[i] < 0]
    assert len(roots)==2
    roots.sort()
    Nr = NN/xx
    I = integrate(xx, Nr, roots[0], roots[1])
    Pi0 = 2 * np.pi**2 / I
    return Pi0


#rotation frequency from Eq.3
def run(l, mm, true_model, assumed_model, f_rot_grid):

    rr, NN = load_Brunt_Vaisala_from_pulse( os.path.join(data_dir, '01_MESA', assumed_model.pulse_name()) )

    Strue = load_gyre_summary( os.path.join(data_dir, '02_GYRE', true_model.freq_name()) )
    Sassumed = load_gyre_summary( os.path.join(data_dir, '02_GYRE', assumed_model.freq_name()) )

    #Isimps = simps(NN/rr, rr)
    #Itrapz = np.trapz(NN/rr, rr)
    #print('Integral of N(r)/r = ', Itrapz, Isimps)
    

    FROT = {1:[], -1:[]}
    nn = [-n for n in range(10, 31)]
    agg, Pi0s = [], []
    #plot_freqs(S, nn, l, model)
    
    for n in nn:
        for m in mm:
            lmn = (l,m,n)
            m_ag = m
            if assumed_model.rot == None: m_ag=0
            f_osc = Sassumed[(l,m_ag,n)]
            # find oscillation cavity
            Pi0 = get_Pi0_precise(rr, NN, f_osc)
            Pi0s.append(Pi0)
            alpha_g = compute_alpha_g(rr, NN, f_osc, assumed_model.get_rot_Hz(), l, m_ag, n)
            agg.append(alpha_g)
            if alpha_g_plot:
                plot_alpha_g(assumed_model.freq_name(), lmn, lambda x: compute_alpha_g(rr, NN, Sassumed[lmn], x, l, m, n) - alpha_g)
                print(lmn, alpha_g)
            try:
                #f_rot_find = get_f_rot_Newton(rr, NN, fstart, Strue[lmn], l, m, -n, alpha_g)
                f_rot_roots = get_f_rot_Brent(rr, NN, f_rot_grid[m], Strue[lmn], l, m, -n, alpha_g)
                f_rot_uHz = [f * uHz for f in f_rot_roots]
                FROT[m].append(f_rot_uHz)
                print(n, 'f_in =', Strue[lmn] * uHz, '[uHz], ', 'f_rot_find =', f_rot_uHz, '[uHz]', 'alpha_g =', alpha_g, 'Pi0 =', Pi0)
            except Exception as ex:
                FROT[m].append([np.nan])
                print(ex)

    fig = Figure('Radial order', 'Rotation frequency [$\\mu$Hz]', grid=False, xticks=even(nn), legend=False)

    for m in mm:
        fig.add_series(nn, prep_series(FROT[m]), style=fmt[m], label='m='+str(m), color='black')

    fig.add_series(nn, [true_model.get_rot_Hz() * uHz for n in nn], style='--', color='black')
    name = 'figures/'+'_'.join([str(x) for x in [true_model, assumed_model, 'l'+str(l), 'm'+''.join([str(m) for m in mm])]])
    noise = '.N' if true_model.noise else ''
    fig.dump(name+noise+'.fig')

    ag_fig = Figure('Radial order','alpha_g', xticks=even(nn), legend=False)
    ag_fig.add_series(nn, agg, style='o-', color='black')
    ag_fig.dump(name + '.AG.fig')

    p0_fig = Figure('Radial order','Pi_0 [s]', xticks=even(nn), legend=False)
    p0_fig.add_series(nn, Pi0s, style='o-', color='black')
    p0_fig.dump(name + '.Pi0.fig')

    if show_fig: fig.show()
    else: fig.clear()

#rotation frequency from Eq.3
def get_f_rot(ll, mm, nn, true_model, assumed_model, f_rot_grid):

    rr, NN = load_Brunt_Vaisala_from_pulse( os.path.join(data_dir, '01_MESA', assumed_model.pulse_name()) )

    Strue = load_gyre_summary( os.path.join(data_dir, '02_GYRE', true_model.freq_name()) )
    Sassumed = load_gyre_summary( os.path.join(data_dir, '02_GYRE', assumed_model.freq_name()) )

    FROT = {}
    
    for l in ll:
        for m in mm:
            for n in nn:
                lmn = (l,m,n)
                m_ag = m
                if assumed_model.rot == None: m_ag=0
                f_osc = Sassumed[(l,m_ag,n)]
                alpha_g = compute_alpha_g(rr, NN, f_osc, assumed_model.get_rot_Hz(), l, m_ag, n)

                f_rot_roots = get_f_rot_Brent(rr, NN, f_rot_grid[m], Strue[lmn], l, m, -n, alpha_g)
                f_rot_uHz = [f * uHz for f in f_rot_roots]
                FROT[lmn] = f_rot_uHz
    return FROT


models = {}
models[('1.5','0.2')] = [Model('1.515', '0.2', '10'), Model('1.5', '0.202', '10'), Model('1.5', '0.2', '9.9'),
                         Model('1.65',  '0.2', '10'), Model('1.5', '0.22',  '10'), Model('1.5', '0.2', '9.0')]
models[('1.5','0.6')] = [Model('1.515', '0.6', '10'), Model('1.5', '0.594', '10'), Model('1.5', '0.6', '9.9'),
                         Model('1.65',  '0.6', '10'), Model('1.5', '0.54',  '10'), Model('1.5', '0.6', '9.0')]
models[('3.0','0.2')] = [Model('3.03', '0.2', '10'), Model('3.0', '0.202', '10'), Model('3.0', '0.2', '9.9'), 
                         Model('3.3',  '0.2', '10'), Model('3.0', '0.22',  '10'), Model('3.0', '0.2', '9.0')]
models[('3.0','0.6')] = [Model('3.03', '0.6', '10'), Model('3.0', '0.594', '10'), Model('3.0', '0.6', '9.9'), 
                         Model('3.3',  '0.6', '10'), Model('3.0', '0.54',  '10'), Model('3.0', '0.6', '9.0')]


def make_table(nn):
    def check_pulse(fn):
        path = os.path.join(data_dir, '01_MESA', fn)
        if not os.path.isfile(path): print(path)

    def check_freq(fn):
        path = os.path.join(data_dir, '02_GYRE', fn)
        if os.path.isfile(path):
            print('OK', fn)
        else:
            print('pulse_'+fn[:-4].replace('.omega.','.mesa.omega.'), ', ')

    def format_pcnt(d, f_true):
        a = []
        for k in d:
            a.append( ' %.3f '%(100* (d[k][0] - f_true)/f_true) )
        return ' '.join(a)

    f_rot_grid = {}
    f_rot_grid[-1] = np.linspace(5/uHz, 15/uHz, 75)
    f_rot_grid[1] = np.linspace(8/uHz, 12/uHz, 10)

    ll,mm = [1,2],[1]

    for k in models:
        Mtrue = Model(k[0], k[1], '10')
        for Massumed in models[k]:
            try:
                frot = get_f_rot(ll, mm, nn, Mtrue, Massumed, f_rot_grid)
                print(Mtrue, Massumed, format_pcnt(frot, 10.))
                run(1, mm, Mtrue, Massumed, f_rot_grid)
            except Exception as ex:
                print(Mtrue, Massumed, ex)

def make_BV_freq_plots():

    def load_and_plot(mod, fig):
        rr, NN = load_Brunt_Vaisala_from_pulse( os.path.join(data_dir, '01_MESA', mod.pulse_name()) )
        fig.add_series(rr, NN, label=mod.label())

    for k in models:
        fig = Figure('Radius [R_star]','Brunt-Vaisala frequency [Hz]', grid=False)

        Mtrue = Model(k[0], k[1], '10')
        load_and_plot(Mtrue, fig)
        for Massumed in models[k]:
            if Massumed.rot != Mtrue.rot: continue
            load_and_plot(Massumed, fig)
        fig.savefig('figures/BV/' + str(Mtrue) + '.pdf')
        fig.dump('figures/BV/' + str(Mtrue) + '.fig')
        fig.show()

if __name__=='__main__x':

    #make_table([-30, -20])
    make_BV_freq_plots()


if __name__=='__main__x':

    Mtrue = Model('1.5', '0.2', '10')
    Mtrue.noise = True
    Massumed = Model('1.515', '0.2', '10')

    f_rot_grid = {}
    f_rot_grid[-1] = np.linspace(5/uHz, 15/uHz, 75)
    f_rot_grid[1] = np.linspace(8/uHz, 12/uHz, 10)

    run(1, [1], Mtrue, Massumed, f_rot_grid)







