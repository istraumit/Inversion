import os, sys
import numpy as np
import corner

root = sys.argv[1]

path = os.path.join(root, 'samples.npy')
samples = np.load(path)
fig = corner.corner(samples)
fig.savefig(os.path.join(root,'triangle.png'))



