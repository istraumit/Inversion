import math
import numpy as np
import matplotlib.pyplot as plt

sample = 10 + np.random.randn(1000)
print(np.mean(sample), np.std(sample))

sample2 = 10*sample
print(np.mean(sample2), np.std(sample2))

plt.hist(sample, 50)
plt.show()









