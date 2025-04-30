using LinearAlgebra
using Distributions
using Statistics
using MLJBase

# -------------------------------------------------------------
# condición inicial
θ= 10 * rand(Normal(), 2, 1)
# -------------------------------------------------------------
# parámetros del optimizador
α = 0.95
η = 0.9
μ = 0.9
τ = 0.8
batch_size = 300
# -------------------------------------------------------------
# parámetros de la función objetivo
κ = 0.01
# -------------------------------------------------------------
# máximo número de iteraciones
nIter = 500
# -------------------------------------------------------------
# datos iniciales
n_samples = 500
Xt, yt = make_regression(n_samples, 1; noise=0.5, rng=10)
# -------------------------------------------------------------
# verificación de función y gradiente
include("functions.jl")
println("Verificación de vector inicial y evaluación de función y gradiente")
f = func_exp(θ, Xt.x1, yt, κ)
g = grad_exp(θ, Xt.x1, yt, κ)
println("θ =$(θ): size=$(size(θ)), typeof=$(typeof(θ))")
println("fx=$(f): size=$(size(f)), typeof=$(typeof(f))")
println("∇ =$(g): size=$(size(g)), typeof=$(typeof(g))")

f_params = (Xt=Xt.x1, yt=yt, κ=κ)
println("Optimización por métodos de primer orden")
# First order methods
include("methods_1order.jl")
Θ_GD = GD(θ, grad_exp, f_params, nIter, α)
println("GD, Inicio: $(Θ_GD[1:end,1]), Fin: $(Θ_GD[1:end,end]), Pasos: $(size(Θ_GD, 2)-1), f(x)=$(func_exp(Θ_GD[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_GD[1:end,end], f_params...))")

Θ_SGD = SGD(θ, grad_exp, f_params, nIter, α, batch_size)
println("SGD, Inicio: $(Θ_SGD[1:end,1]), Fin: $(Θ_SGD[1:end,end]), Pasos: $(size(Θ_SGD, 2)-1), f(x)=$(func_exp(Θ_SGD[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_SGD[1:end,end], f_params...))")

Θ_MGD = MGD(θ, grad_exp, f_params, nIter, α, η)
println("MGD, Inicio: $(Θ_MGD[1:end,1]), Fin: $(Θ_MGD[1:end,end]), Pasos: $(size(Θ_MGD, 2)-1), f(x)=$(func_exp(Θ_MGD[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_MGD[1:end,end], f_params...))")

Θ_NAG = NAG(θ, grad_exp, f_params, nIter, α, η)
println("NAG, Inicio: $(Θ_NAG[1:end,1]), Fin: $(Θ_NAG[1:end,end]), Pasos: $(size(Θ_NAG, 2)-1), f(x)=$(func_exp(Θ_NAG[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_NAG[1:end,end], f_params...))")

#α = 0.7
#Θ_ADADELTA = ADADELTA(θ, grad_exp, f_params, nIter, α, η)
#println("ADADELTA, Inicio: $(Θ_ADADELTA[1:end,1]), Fin: $(Θ_ADADELTA[1:end,end]), Pasos: $(size(Θ_ADADELTA, 2)-1), f(x)=$(func_exp(Θ_ADADELTA[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_ADADELTA[1:end,end], f_params...))")

α = 0.95
η1 = 0.9
η2 = 0.999
θ_ADAM = ADAM(θ, grad_exp, f_params, nIter, α, η1, η2)
println("ADAM, Inicio: $(θ_ADAM[1:end,1]), Fin: $(θ_ADAM[1:end,end]), Pasos: $(size(θ_ADAM, 2)-1), f(x)=$(func_exp(θ_ADAM[1:end,end], f_params...)), ∇(x)=$(grad_exp(θ_ADAM[1:end,end], f_params...))")

println("Done!")

include("methods_2order.jl")
println("Optimización por métodos de segundo orden")
# Second order methods
Θ_SR1TR = LSR1TR(θ, func_exp, grad_exp, f_params, nIter, α, batch_size)
println("SR1-TR, Inicio: $(Θ_SR1TR[1:end,1]), Fin: $(Θ_SR1TR[1:end,end]), Pasos: $(size(Θ_SR1TR, 2)-1), f(x)=$(func_exp(Θ_SR1TR[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_SR1TR[1:end,end], f_params...))")

Θ_SSR1TR = LSSR1TR(θ, func_exp, grad_exp, f_params, nIter, α, μ, batch_size)
println("SSR1-TR, Inicio: $(Θ_SR1TR[1:end,1]), Fin: $(Θ_SR1TR[1:end,end]), Pasos: $(size(Θ_SR1TR, 2)-1), f(x)=$(func_exp(Θ_SR1TR[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_SR1TR[1:end,end], f_params...))")

Θ_MBSR1TR = MBSR1TR(θ, func, grad, f_params, nIter, α, μ, τ, batch_size)
println("MBSTR1-TR, Inicio: $(Θ_MBSR1TR[1:end,1]), Fin: $(Θ_MBSR1TR[1:end,end]), Pasos: $(size(Θ_MBSR1TR, 2)-1), f(x)=$(func_exp(Θ_MBSR1TR[1:end,end], f_params...)), ∇(x)=$(grad_exp(Θ_MBSR1TR[1:end,end], f_params...))")
#=
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
=#