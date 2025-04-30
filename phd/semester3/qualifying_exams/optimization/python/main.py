import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from functions import rosenbrock2d
from methods import steepest_descend, bfgs, lbfgs, SR1_TR

rosenbrock2d_grad = grad(rosenbrock2d)
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

if False:
    x0 = np.array([-1.2,1])
    history = steepest_descend(rosenbrock2d, rosenbrock2d_grad, x0, eps=1e-3, K=10000)
    print(f"El mínimo de la función es: {history[-1]}, en {len(history)} pasos")

    x0 = np.array([-1.2,1])
    history = bfgs(rosenbrock2d, rosenbrock2d_grad, x0, np.eye(len(x0)), eps=1e-5, K=10000)
    print(f"El mínimo de la función es: {history[-1]}, en {len(history)} pasos")

    x0 = np.array([-1.2,1])
    history = lbfgs(rosenbrock2d, rosenbrock2d_grad, x0, 5, eps=1e-5, K=10000)
    print(f"El mínimo de la función es: {history[-1]}, en {len(history)} pasos")

x0 = np.array([-1.2,1])
history = SR1_TR(rosenbrock2d, rosenbrock2d_grad, x0, np.eye(len(x0)), delta_0=1, eps=1e-5, K=10000)
print(f"El mínimo de la función es: {history[-1]}, en {len(history)} pasos")
