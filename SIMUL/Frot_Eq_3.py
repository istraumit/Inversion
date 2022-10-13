import os
from Equation3 import *
from Figure import Figure

save = True

fmt = {1:'o-', -1:'s-'}

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
    xx = np.linspace(0, 1.e-4, 50)
    yy = [ag_func(x) for x in xx]
    fig = Figure('f_rot [microHz]', 'F', title='(l,m,n) = '+str(lmn), legend=False)

    fig.add_series(1.e6*xx, yy)

    fig.add_lines('v', [1.e-6], style='--', color='black')
    fig.add_lines('h', [0.0], style='-', color='black')

    fig.dump('figures/roots/'+model+'_'+ str(lmn) +'.fig')

#rotation frequency from Eq.3 with alpha_g from m=0
def run(l, model, offset):
    data_dir = '/home/elwood/Documents/Inversion/DATA'
    if offset:
        pulse_path = os.path.join(data_dir, '01_MESA', 'OFF_MODELS', 'pulse_' + off_models[model] + '.mesa')
        freq_path_nonrot = os.path.join(data_dir, '02_GYRE', 'OFF', off_models[model])
    else:
        pulse_path = os.path.join(data_dir, '01_MESA', 'pulse_' + model + '.mesa')
        freq_path_nonrot = os.path.join(data_dir, '02_GYRE', model)

    freq_path_rot = os.path.join(data_dir, '02_GYRE', model + '.omega.const')
    rr, NN = load_Brunt_Vaisala_from_pulse(pulse_path)
    S = load_gyre_summary(freq_path_rot)
    Snorot = load_gyre_summary(freq_path_nonrot)
    fstart = 0.0
    FROT = {1:[], -1:[]}
    nn = [-n for n in range(10, 31)]
    #plot_freqs(S, nn, l, model)
    
    for n in nn:
        alpha_g = compute_alpha_g(rr, NN, Snorot[(l,0,n)], 0, l, 0, n)
        for m in [-1,1]:
            if False:
                plot_alpha_g(model, (l,m,n), lambda x: compute_alpha_g(rr, NN, S[(l,m,n)], x, l, m, n)-alpha_g)
                print(l,m,n)
            try:
                f_rot_find = get_f_rot_Newton(rr, NN, fstart, S[(l,m,n)], l, m, -n, alpha_g)
                FROT[m].append(f_rot_find*1.e9)
            except:
                FROT[m].append(np.nan)

    fig = Figure('Radial order', 'Rotation frequency [nHz]', grid=False, xticks=even(nn))

    for m in [-1,1]:
        fig.add_series(nn, FROT[m], style=fmt[m], label='m='+str(m), color='black')

    fig.add_series(nn, [1.e3 for n in nn], style='--', color='black')
    if save:
        if offset:
            name = 'figures/'+model+'_OFF'+off_models[model]+'_l'+str(l)+'_Frot'
        else:
            name = 'figures/'+model+'_l'+str(l)+'_Frot'
        fig.savefig(name+'.pdf')
        fig.dump(name+'.fig')
    else:
        fig.show()

if __name__=='__main__':
    ll = [1,2]
    MM = ['M1.5_XC0.2', 'M1.5_XC0.6', 'M3.0_XC0.2', 'M3.0_XC0.6']

    off_models = {
    MM[0]:'M1.5_XC0.16',
    MM[1]:'M1.5_XC0.48',
    MM[2]:'M3.0_XC0.16',
    MM[3]:'M3.0_XC0.48'}

    off_models = {
    MM[0]:'M1.35_XC0.2',
    MM[1]:'M1.35_XC0.6',
    MM[2]:'M2.7_XC0.2',
    MM[3]:'M2.7_XC0.6'}

    off_models = {
    MM[0]:'M1.65_XC0.2',
    MM[1]:'M1.65_XC0.6',
    MM[2]:'M3.3_XC0.2',
    MM[3]:'M3.3_XC0.6'}

    for l in ll:
        for M in MM:
            print(l, M)
            run(l, M, offset=True)





