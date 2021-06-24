import sys
import numpy as np
import matplotlib.pyplot as plt
from DataCube import *
from Estimator import *
from Visualizer import *

A = np.load(sys.argv[1])

DC = DataCube(['M', 'Z', 'Xc'])
DC.load_from_array(A)

est = Estimator(DC)
vis = Visualizer(est)



vis.plot_P_slices()
plt.show()

#vis.plot_CDFs()
#plt.show()


PAR = est.get_parameter_estimates(['%.2f', '%.4f', '%.3f'])
print('-'*25)
for p in PAR: print(p, ':', PAR[p])
print('-'*25)
    
