import numpy as np
from matplotlib import pyplot as plt

# 2.22 Función de Rosenbrock
# Nocedal & Wright (2006) Numerical Optimization, pp. 27
def rosenbrock2d(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradiente de la Función de Rosenbrock
def rosenbrock2d_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
         200     *(x[1] - x[0]**2)
    ])

def quadratic(A, b, c):
    def eval(x):
        return (0.5*(np.matmul(x.T, np.matmul(A, x))) - np.matmul(b.T, x) + c)
    return eval

def grad_quadratic(A, b):
    def eval(x):
        return np.matmul(A, x) - b
    return eval

def lambda_quadratic(A):
    def eval(g):
        return (np.matmul(g.T, g))/(np.matmul(g.T, np.matmul(A, g)))
    return eval

if __name__ == '__main__':
    x1 = np.linspace(- 0.5, + 0.5, 30)
    x2 = np.linspace(- 0.5, + 0.5, 30)
    X1, X2 = np.meshgrid(x1, x2)
    Z = rosenbrock2d([X1, X2])
    plt.figure()
    plt.title('Rosenbrock')
    plt.contourf(X1, X2, Z, 30, cmap='jet')
    plt.colorbar()
    plt.xlabel('$x_1$');
    plt.ylabel('$x_2$')
    plt.show()
