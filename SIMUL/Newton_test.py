import sys


def dfdx(opt, x):
    dx = 1.e-10
    f0 = opt(x)
    f1 = opt(x + dx)
    df = f1 - f0
    return f0, df/dx

def root(x_start):

    def opt(x):
        return x**2 - 1

    x0 = x_start
    print(x0)
    while True:
        f0, deriv = dfdx(opt, x0)
        x1 = x0 - f0/deriv
        if abs(x0-x1)==0.0: return x1
        x0 = x1
        print(x1)


if __name__=='__main__':
    start = float(sys.argv[1])
    print(root(start))
