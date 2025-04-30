# Definición de librerías
using LinearAlgebra
using Distributions
using Statistics
using MLJBase
using Random
using Plots

function GD(θ, grad, f_params, nIter, α)    
    #=
    Descenso de gradiente

    Parámetros
    -----------
    θ0        :   condicion inicial
    grad      :   función que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                     nIter = gd_params[0] número de iteraciones
                     alpha = gd_params[1] tamaño de paso alpha

    f_params  :   lista de parametros para la funcion objetivo
                     kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                     X     = f_params['X'] Variable independiente
                     y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Θ     :   trayectoria de los parametros    
    =#
    Θ = θ
    for iter in 1:nIter
        p = grad(θ, f_params...)
        θ = θ - α * p
        Θ = hcat(Θ, θ)        
    end
    return Θ
end

#= Descenso de gradiente estocástico =#
function SGD(θ, grad, f_params, nIter, α, batch_size)
    Θ = θ
    for iter in 1:nIter
        # Obtenemos la muestra de tamaño (batch size)
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        yt_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
        p = grad(θ, f_params_sample...)
        θ = θ - α * p
        Θ = hcat(Θ, θ)
    end
    return Θ
end

#= Descenso de gradiente con momento (inercia) =#
function MGD(θ, grad, f_params, nIter, α, η)
    p_old = zeros(size(θ))
    Θ = θ
    for iter in 1:nIter
        g = grad(θ, f_params...)
        p = g + η * p_old
        θ = θ - α * p
        p_old = p
        Θ = hcat(Θ, θ)
    end
    return Θ
end

#= Descenso acelerado de Nesterov =#
function NAG(θ, grad, f_params, nIter, α, η)
    p = zeros(size(θ))
    Θ = θ

    for iter in 1:nIter
        pre_θ = θ - 2.0 * α * p
        g = grad(pre_θ, f_params...)
        p = g + η * p
        θ = θ - α * p
        Θ = hcat(Θ, θ)
    end
    return Θ
end

#= Descenso de Gradiente Adaptable (ADADELTA) =#
function ADADELTA(θ, grad, f_params, nIter, α, η)
    ϵ = 1e-8
    G = zeros(size(θ))
    g = zeros(size(θ))
    Θ = θ

    for iter in 1:nIter
        g = grad(θ, f_params...)
        G = η * g .^ 2 + (1 - η) * G
        p = 1.0 ./ (sqrt.(G) .+ ϵ) .* g
        θ = θ - α * p
        Θ = hcat(Θ, θ)
    end
    return Θ
end

#= Descenso de Gradiente Adaptable con Momentum(ADAM) =#
function ADAM(θ, grad, f_params, nIter, α, η1, η2)
    ϵ = 1e-8
    p = zeros(size(θ))
    v = 0.0
    Θ = θ
    η1_t = η1
    η2_t = η2
    for iter in 1:nIter
        g = grad(θ, f_params...)
        p = η1 * p .+ (1.0 - η1) .* g
        v = η2 * v .+ (1.0 - η2) .* (g .^ 2)
        θ = θ - α * p ./ (sqrt.(v) .+ ϵ)
        η1_t *= η
        η2_t *= η
        Θ = hcat(Θ, θ)
    end
    return Θ
end