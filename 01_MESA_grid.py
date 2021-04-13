import sys, os
import shutil
import numpy as np
from config import parse_conf
from inlist import prep_inlist
from utils import run

def get_history_value(param_name):
    fn = 'LOGS/history.data'
    data = np.loadtxt(fn, skiprows=6)
    n = 0
    with open(fn) as f:
        for line in f:
            n += 1
            if n==6:
                header = line.split()
                break
            
    ind = header.index(param_name)
    v = data[-1,ind]
    return v


opt = parse_conf()

stage_dir = opt['MESA_stage_dir']
data_dir = opt['DATA_dir']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)

MESA_dir = opt['MESA_dir']
os.chdir(MESA_dir)

template = opt['MESA_inlist_template']
subst = {}

N_grid = int(opt['MESA_N_samples'])
mass_distr = [float(x) for x in opt['MESA_mass_distr']]
Xc_distr = [float(x) for x in opt['MESA_Xc_distr']]

for i in range(N_grid):
    mass = '%.4f'%(mass_distr[0] + np.random.randn() * mass_distr[1])
    Xc = '%.5f'%(Xc_distr[0] + np.random.randn() * Xc_distr[1])

    subst['initial_mass'] = mass
    subst['xa_central_lower_limit(1)'] = Xc

    prep_inlist(template, 'inlist_project', subst)
    
    #run('./rn')
        
    prefix = 'M'+mass+'_Xc'+Xc

    pulse_name = 'pulse_' + prefix + '.mesa'
    history_name = 'history_' + prefix + '.data'

    run('cp pulse.mesa ' + os.path.join(out_dir, pulse_name))
    run('cp LOGS/history.data ' + os.path.join(out_dir, history_name))

    print('MESA run complete for M =', mass, '  Xc =', Xc)




