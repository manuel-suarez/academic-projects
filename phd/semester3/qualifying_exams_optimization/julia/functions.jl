# -------------------------------------------------------------
function func_quadratic(θ, Xt, yt)
    #=
    Funcion de costo
        sum_i (theta@x[i]-y[i])**2
    Parámetros:
    -----------
    θ   : variables del modelo
    Xt  : variables observadas
    yt  : variables a predecir
    =#
    return sum((θ[1] * Xt + θ[2] -yt).^2)
end

function grad_quadratic(θ, Xt, yt)
    #=
    Gradiente de la funcion de costo
        sum_i (theta@x[i]-y[i])**2
    Parámetros:
    -----------
    θ   : variables del modelo
    Xt  : variables observadas
    yt  : variables a predecir
    =#
    err = θ[1] * Xt + θ[2] - y
    ∂1 = err
    ∂2 = Xt * ∂1
    ∇ = [∂1, ∂2]
    return sum(∇)
end

# -------------------------------------------------------------
function func_exp(θ, Xt, yt, κ)
    #=
    Funcion de costo
        sum_i 1-exp(-κ(θ@x[i]-y[i])^2)
    =#
    err = Xt*θ[1] .+ θ[2] - yt
    return sum(1 .- exp.(-κ*err.^2))
end

function grad_exp(θ, Xt, yt, κ)
    #=
    Gradiente de la funcion de costo
        sum_i 1-exp(-κ(θ@x[i]-y[i])^2)
    =#
    err = Xt*θ[1] .+ θ[2] - yt
    ∂1 = err .* exp.(-κ*err.^2)
    ∂2 = Xt .* ∂1
    gradient = zeros(2, 1)
    gradient[1] = mean(∂1)
    gradient[2] = mean(∂2)
    return gradient
end
