import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def obj(x): #rosenbrock
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_grad(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H

'''
# Auckley
f = lambda x: 20*np.exp(0.2*x) - np.exp(np.cos(2*np.pi*x)) + 20 + np.exp(1)
grad = lambda x: 10*np.exp(0.2*x) + np.exp(np.cos(2*np.pi*x))*np.sin(2*np.pi*x)*2*np.pi
hess = lambda x: 2*np.exp(0.2*x) - (2*np.pi)**2*np.exp(np.cos(2*np.pi*x))*(np.sin(2*np.pi*x))**2 + (2*np.pi)**2*np.exp(np.cos(2*np.pi*x))*np.cos(2*np.pi*x)
'''

# Gramacy/Lee
f = lambda z: np.sin(10*np.pi*z)/(2*z) + (z-1)**4
grad = lambda z: (2*z*np.cos(10*np.pi*z)*10*np.pi - 2*np.sin(10*np.pi*z))/(4*z**2) + 4*(z-1)**3
hess = lambda x: -(50*np.pi**2*np.sin(10*np.pi*x))/x+np.sin(10*np.pi*x)/x**3-(10*np.pi*np.cos(10*np.pi*x))/x**2+12*(x-1)**2

for run in range(10):
    J = 10
    eps = 0.001
    x0 = np.random.uniform(-32,32)

    S_in = np.zeros(J)
    y_in = np.zeros(J)
    x_in = np.zeros(J)
    x_in[0] = x0
    S_in[0] = 100000000
    y_in[0] = f(x_in[0])
    j = 0
    k = 1
    R = []
    R.append(0)
    R.append(1)


    while True:
        if j < J:
            x_in[j+1] = opt.minimize(f, x_in[j], method='BFGS', jac=grad, hess=hess,
                               options={'xtol': 1e-8, 'disp': False}).x
            y_in[j+1] = f(x_in[j+1])
            j = j + 1
            S_in[j] = (y_in[R[k-1]] - y_in[j])/(j - R[k-1])
            if y_in[R[k-1]] > y_in[j] and j > 1:
                k = k + 1
                R.append(j)
                S_in_overall = (y_in[R[1]] - y_in[R[k]])/(R[k] - R[1])
                if S_in_overall <= eps:
                    break
            else:
                continue
        else:
            break

    print(j)
    print(k)
    print()
    '''
    for i in range(j+1):
        x = x_in[i]
        y = y_in[i]
        plt.plot(x, y, 'bo')
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=10)

    plt.scatter(x_in, y_in)
    plt.show()
    '''
    plt.scatter(x_in, y_in)
    plt.savefig("plot" + str(run) + ".png")

# there HAS to be a cleaner way to write this...
