import os, sys
import numpy as np
import subprocess as sp

root_dir = sys.argv[1]

tri_fn = 'triangle.png'
smp_fn = 'samples.npy'

for root, subdirs, files in os.walk(root_dir):
    if os.path.isfile(os.path.join(root, tri_fn)): continue
    if os.path.isfile(os.path.join(root, smp_fn)):
        try:
            o = sp.check_output('python make_corner.py ' + root, shell=True, stderr=sp.STDOUT)
            print(root)
        except sp.CalledProcessError as err:
            print(err.output)
        else:
            pass



