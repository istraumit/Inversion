from math import factorial, pi, sqrt, cos
import cmath
import numpy as np
import matplotlib.pyplot as plt

def c2lm(l, m):
    return (2*l+1)*factorial(l-m)/(4*pi*factorial(l+m))

def P11(x):
    return -sqrt(1 - x**2)

def Y11(theta, phi):
    clm = sqrt(c2lm(1,1))
    Y = -clm * P11(cos(theta)) * cmath.exp(1j*phi)
    return Y.real

if __name__=='__main__':

    print(sqrt(c2lm(1,1)))
    exit()

    xx = np.linspace(-pi, pi, 100)
    rr1 = [Y11(0.5*pi, x) for x in xx]
    rr2 = [Y11(x, 0) for x in xx]

    plt.polar(xx, rr1)
    plt.show()

    plt.polar(xx, rr2)
    plt.show()


