import numpy as np
from scipy.integrate import simps
from bisect import bisect


def differentiate(xi_r, xi, rr):
    deg = 2
    res = []
    for r in rr:
        i = bisect(xi_r, r)
        if i<2: i=2
        if i>len(xi)-3: i = len(xi)-3
        xi_slice = xi[i-2:i+3]
        xi_r_slice = xi_r[i-2:i+3]
        pfit = np.polynomial.polynomial.Polynomial.fit(xi_r_slice, xi_slice, deg)
        coef = pfit.convert().coef
        deriv = np.sum([coef[n] * n * r**(n-1) for n in range(1, deg+1)])
        res.append(deriv)
    return np.array(res)


def integrate(xx, yy, a, b):
    """
    a,b: integration limits
    """
    assert a < b
    assert a >= xx[0]
    assert b <= xx[-1]

    y_edges = np.interp([a,b], xx, yy)

    i_start = bisect(xx, a)
    i_end = bisect(xx, b)

    x_slice = [a] + list(xx[i_start:i_end]) + [b]
    y_slice = [y_edges[0]] + list(yy[i_start:i_end]) + [y_edges[1]]

    I = simps(y_slice, x_slice)
    return I


if __name__=='__main__':
    import matplotlib.pyplot as plt

    xx = [-np.pi] + list(2*np.pi*np.random.rand(100) - np.pi) + [np.pi]
    xx.sort()
    xx = np.array(xx)

    yy = np.sin(xx)
    a = -3
    b = 3

    plt.plot(xx, yy, '.-')
    print('Integration test on unevenly spaced grid')
    print('\int{-3}{3} sin(x) =', integrate(xx, yy, a, b))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(a)
    plt.axvline(b)
    plt.show()

    xx_new = np.linspace(-np.pi, np.pi, 100)
    exact = np.cos(xx_new)

    diff = differentiate(xx, yy, xx_new)

    plt.plot(xx_new, diff, '.-', label='Numerical')
    plt.plot(xx_new, exact, label='Exact')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()









