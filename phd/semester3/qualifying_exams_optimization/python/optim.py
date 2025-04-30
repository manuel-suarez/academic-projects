import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from methods_1order import *
from methods_2order import L_SR1_TR

n_samples = 500
X, y = datasets.make_regression(n_samples=n_samples,
                                n_features=1,
                                n_informative=2,
                                noise=5,
                                random_state=0) #2)
n_outliers=100
X[:n_outliers], y[:n_outliers] = datasets.make_regression(n_samples=n_outliers,
                                n_features=1,
                                n_informative=2,
                                noise=2,
                                random_state=61)
y=np.expand_dims(y,axis=1)
plt.scatter(X[:],y[:], marker='.')


# -------------------------------------------------------------
def func_quadratic(theta, f_params):
    '''
    Funcion de costo
            sum_i (theta@x[i]-y[i])**2
    '''
    X = f_params['X']
    y = f_params['y']
    return np.sum((theta[0] * X + theta[1] -y)**2)

def grad_quadratic(theta, f_params):
    '''
    Gradiente de la funcion de costo
           sum_i (theta@x[i]-y[i])**2
    '''
    X = f_params['X']
    y = f_params['y']

    err = theta[0] * X + theta[1] - y
    partial0 = err
    partial1 = X * partial0
    gradient = np.concatenate((partial1, partial0), axis=1)
    print(gradient.shape)
    return np.sum(gradient, axis=1)


# -------------------------------------------------------------
def func_exp(theta, f_params):
    kappa = f_params['kappa']
    X = f_params['X']
    y = f_params['y']
    err = theta[0] * X + theta[1] - y
    return np.sum(1 - np.exp(-kappa * err ** 2))

def grad_exp(theta, f_params):
    '''
    Gradiente de la funcion de costo
           sum_i 1-exp(-k(theta@x[i]-y[i])**2)
    '''
    kappa = f_params['kappa']
    X = f_params['X']
    y = f_params['y']
    err = theta[0] * X + theta[1] - y
    partial0 = err * np.exp(-kappa * err ** 2)
    partial1 = X * partial0
    gradient = np.concatenate((partial1, partial0), axis=1)
    return np.mean(gradient, axis=0)
# -------------------------------------------------------------
# condición inicial
theta=10*np.random.normal(size=2)
#theta= [-0.61752689 -0.76804482]

# parámetros del algoritmo
gd_params = {'alpha'          : 0.95,
             'alphaADADELTA'  : 0.7,
             'alphaADAM'      : 0.95,
             'nIter'          : 300,
             'batch_size'     : 100,
             'mem_size'       : 20,
             'delta_0'        : 1,
             'gamma_0'        : 1,
             'eps'            : 1e-5,
             'eta'            : 0.9,
             'eta1'           : 0.9,
             'eta2'           : 0.999}

# parámetros de la función objetivo
f_params={'kappa' : 0.01,
          'X'     : X ,
          'y'     : y}

# Second order methods
ThetaLSR1TR = L_SR1_TR(theta_0=theta, func=func_exp, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('L-SR1-TR, Inicio:', theta, '-> Fin:', ThetaLSR1TR[-1,:])

# First order methods
ThetaGD = GD(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('GD, Inicio:', theta,'-> Fin:', ThetaGD[-1,:])
exit(0)

ThetaSGD = SGD(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('SGD, Inicio:', theta,'-> Fin:', ThetaSGD[-1,:])

ThetaMGD = MGD(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('MGD, Inicio:', theta,'-> Fin:', ThetaMGD[-1,:])

ThetaNAG = NAG(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('NAG, Inicio:', theta,'-> Fin:', ThetaMGD[-1,:])

ThetaADADELTA = ADADELTA(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('ADADELTA, Inicio:', theta,'-> Fin:', ThetaADADELTA[-1,:])

ThetaADAM = ADAM(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('ADAM, Inicio:', theta,'-> Fin:', ThetaADAM[-1,:])

Tmax=100
plt.figure(figsize=(10,10))

plt.subplot(211)
plt.plot(ThetaNAG[:Tmax], '.')
plt.title('NAG')

plt.subplot(212)
plt.plot(ThetaADAM[:Tmax], '.')
plt.title('ADAM')

plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 14

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
nIter=np.expand_dims(np.arange(ThetaGD.shape[0]),1)
Tmax=200
ax.plot(ThetaGD[:Tmax,0],  ThetaGD [:Tmax,1], nIter[:Tmax,0], label='GD')
ax.plot(ThetaSGD[:Tmax,0], ThetaSGD[:Tmax,1], nIter[:Tmax,0], label='SGD')
ax.plot(ThetaMGD[:Tmax,0], ThetaMGD[:Tmax,1], nIter[:Tmax,0], label='MGD')
ax.plot(ThetaNAG[:Tmax,0], ThetaNAG[:Tmax,1], nIter[:Tmax,0], label='NAG')
ax.plot(ThetaADADELTA[:Tmax,0], ThetaADADELTA[:Tmax,1], nIter[:Tmax,0], label='ADADELTA')
ax.plot(ThetaADAM[:Tmax,0], ThetaADAM[:Tmax,1], nIter[:Tmax,0], label='ADAM')
ax.legend()
ax.set_title(r'Trayectorias los parámetros calculados con distintos algoritmos')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_0$')
ax.set_zlabel('Iteración')
plt.show()