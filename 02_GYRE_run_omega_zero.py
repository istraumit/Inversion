import sys, os
import shutil
import numpy as np
import subprocess as sp
from config import parse_conf
from utils import run


opt = parse_conf()

MESA_stage_dir = opt['MESA_stage_dir']
data_dir = opt['DATA_dir']
stage_dir = opt['GYRE_stage_dir_omega_zero']
out_dir = os.path.join(data_dir, stage_dir)
os.makedirs(out_dir, exist_ok=True)

GYRE_dir = opt['GYRE_dir']
os.chdir(GYRE_dir)

template = opt['GYRE_inlist_template']

MESA_grid_dir = os.path.join(data_dir, MESA_stage_dir)

models = [fn for fn in os.listdir(MESA_grid_dir) if fn.endswith('.mesa')]

for model in models:
    model_path = os.path.join(MESA_grid_dir, model)

    run('cp ' + model_path + ' pulse.mesa')

    run('./gyre ' + template)

    out_run_dir = os.path.join(out_dir, model[:-5])
    run('mkdir ' + out_run_dir)
    run('mv mode_* ' + out_run_dir)
    run('mv summary.txt ' + out_run_dir)

    print(model, 'complete')




