# Implementación de métodos de segundo orden
using LinearAlgebra
using Statistics
using MLJBase
include("TSsubproblem_solver_OBS.jl")

function LSR1TR(θ, func, grad, f_params, nIter, α, batch_size)
    S           = [];
    Y           = [];
    Ψ           = [];
    Minv        = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;

    skip        = 0;
    k           = 0;
    epoch_k     = 0;
    Θ = θ
    
    for iter in 1:nIter
        #= println("=======================> Iteración $(iter), k=$(k) <==================="); =#
        f           = func(θ, f_params...)
        g           = grad(θ, f_params...)
        llgll       = norm(g)

        # Trust-Region subproblem
        if k == 0 || size(S,2) == 0
            p       = -delta*(g/llgll)
            Bp      =  γ*p
        else
            p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
            Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
        end

        Q_p         = (p'*(g + 0.5*Bp))[]
        llpll       = norm(p)

        # Actualización del vector de pesos
        θ_new       = θ + p
        f_new       = func(θ_new, f_params...)
        g_new       = grad(θ_new, f_params...)

        # Vectores de curvatura (s,y)
        s           = p
        y           = g_new - g
        
        # Reduction ratio
        ρ           = (f_new - f) / Q_p

        # Condición de aceptación del incremento
        θ = θ_new
        f = f_new
        g = g_new
        Θ = hcat(Θ, θ)

        # Condición de paro
        if llgll < tol
            println("Condición de paro")
            return
        end

        # Actualización del radio de la región de confianza
        if ρ > 0.75
            if norm(p) ≤ 0.8*delta
                delta_new = delta
            else
                delta_new = 2*delta
            end
        elseif (0.1 ≤ ρ && ρ ≤ 0.75)
            delta_new = delta
        else
            delta_new = 0.5*delta
        end
        delta = delta_new

        # Condición de Actualización
        y_Bs        = y - Bp
        if abs((s'*y_Bs)[]) > 1e-8*llpll*norm(y_Bs)
            #println("Curvature pairs")
            if k == 0
                S = s
                Y = y
            else
                S = [S s]
                Y = [Y y]
            end
            if (size(S,2) > lim_m)
                S = S[:, 2:end]
                Y = Y[:, 2:end]
            end

            if size(S,2) == 0
                println("S is empty!")
            end

            while (size(S, 2) > 0)                
                SY      = S'*Y
                SS      = S'*S
                if size(SY) == ()
                    LDLt = SY
                else
                    LDLt = tril(SY) + tril(SY,-1)'
                end
                eig_val = eigen(LDLt,SS).values
                λHat_min= minimum(eig_val)
                if λHat_min > 0
                    γ   = max(0.5*λHat_min, 1e-6)
                else
                    γ   = min(1.5*λHat_min, -1e-6)
                end

                Minv    = (LDLt - γ*SS)
                Ψ       = Y - γ*S

                if size(Ψ,2) == rank(Ψ) && rank(Minv) == size(Minv,2)
                    break
                else
                    #println("Psi is NOT full column rank! or M is not invertible!")
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end
            end
        end

        k = k + 1;
    end
    return Θ
end


"""
    LSSR1TR(θ, func, grad, f_params, nIter, α, batch_size)

    La implementación del método SR1 con lote estocástico es básicamente una modificación del método
    anterior SR1 con región de confianza pero tomando ahora lotes de manera aleatoria durante el paso
    de la optimización y utilizar este lote para el cálculo del gradiente mas un factor de ajuste de 
    corrección del mismo
"""
function LSSR1TR(θ, func, grad, f_params, nIter, α, μ, batch_size)
    S           = [];
    Y           = [];
    Ψ           = [];
    Minv        = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;

    skip        = 0;
    k           = 0;
    v_old       = 0;
    epoch_k     = 0;
    Θ           = θ;
    θ_old       = θ;
    
    for iter in 1:nIter
        #= println("=======================> Iteración $(iter), k=$(k) <==================="); =#
        # Al inicio de la iteración determinamos y obtenemos la muestra de tamaño (batch size)
        # sobre el conjunto de entrenamiento, este lote se utilizará para el cálculo del gradiente
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        yt_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
        
        f           = func(θ, f_params_sample...)
        g           = grad(θ, f_params_sample...)
        llgll       = norm(g)

        # Trust-Region subproblem
        if k == 0 || size(S,2) == 0
            p       = -delta*(g/llgll)
            v_old   =  p
            Bp      =  γ*p
        else
            p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
            # Calculamos factor de ajuste sobre el vector de dirección usando el valor de θ en la iteración previa
            v       = μ * v_old + (θ - θ_old)
            v       = μ * min(1.0, delta/norm(v)) * v
            v_old   = v
            # Actualizamos gradiente direccional
            p       = min(1.0, delta/norm(p + v)) * (p + v)
            Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
        end        

        Q_p         = (p'*(g + 0.5*Bp))[]
        llpll       = norm(p)

        # Actualización del vector de pesos
        θ_new       = θ + p
        f_new       = func(θ_new, f_params_sample...)
        g_new       = grad(θ_new, f_params_sample...)

        # Vectores de curvatura (s,y)
        s           = p
        y           = g_new - g
        
        # Reduction ratio
        ρ           = (f_new - f) / Q_p

        # Condición de aceptación del incremento
        θ_old = θ # Para cálculo de factor de ajuste sobre el gradiente direccional
        θ = θ_new
        f = f_new
        g = g_new
        Θ = hcat(Θ, θ)
        # Condición de paro
        if llgll < tol
            println("Condición de paro")
            return
        end

        # Actualización del radio de la región de confianza
        if ρ > 0.75
            if norm(p) ≤ 0.8*delta
                delta_new = delta
            else
                delta_new = 2*delta
            end
        elseif (0.1 ≤ ρ && ρ ≤ 0.75)
            delta_new = delta
        else
            delta_new = 0.5*delta
        end
        delta = delta_new

        # Condición de Actualización
        y_Bs        = y - Bp
        if abs((s'*y_Bs)[]) > 1e-8*llpll*norm(y_Bs)
            #println("Curvature pairs")
            if k == 0
                S = s
                Y = y
            else
                S = [S s]
                Y = [Y y]
            end
            if (size(S,2) > lim_m)
                S = S[:, 2:end]
                Y = Y[:, 2:end]
            end

            if size(S,2) == 0
                println("S is empty!")
            end

            while (size(S, 2) > 0)                
                SY      = S'*Y
                SS      = S'*S
                if size(SY) == ()
                    LDLt = SY
                else
                    LDLt = tril(SY) + tril(SY,-1)'
                end
                eig_val = eigen(LDLt,SS).values
                λHat_min= minimum(eig_val)
                if λHat_min > 0
                    γ   = max(0.5*λHat_min, 1e-6)
                else
                    γ   = min(1.5*λHat_min, -1e-6)
                end

                Minv    = (LDLt - γ*SS)
                Ψ       = Y - γ*S

                if size(Ψ,2) == rank(Ψ) && rank(Minv) == size(Minv,2)
                    break
                else
                    #println("Psi is NOT full column rank! or M is not invertible!")
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end
            end
        end

        k = k + 1;
    end
    return Θ
end

"""
    MB-LSR1-TR(θ, func, grad, f_params, nIter, α, batch_size)

    Finalmente para el algoritmo MB-LSR1 (página 5, Griffin, et. al.) es una modificación del algoritmo SR1-TR con lotes
    pero agregando un factor de corrección del tamaño del lote, el radio de la región de confianza y al gradiente direccional
    así como un factor de tolerancia para hacer un reset completo a los vectores de curvatura, esto debe permitir hacer una
    corrección en el cálculo del gradiente para la búsqueda de la dirección de optimización.
"""
function MBSR1TR(θ, func, grad, f_params, nIter, α, μ, τ, batch_size)
    S           = [];
    Y           = [];
    Ψ           = [];
    Minv        = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;
    ρ_hat       = 1;    # Parámetros de actualización de la región de confianza
    T           = 1;    # Parámetros de actualización de la región de confianza

    skip        = 0;
    k           = 0;
    K           = 50;   # Corrección del tamaño de lote
    v_old       = 0;
    epoch_k     = 0;
    Θ           = θ;
    θ_old       = θ;
    
    for iter in 1:nIter
        #= println("=======================> Iteración $(iter), k=$(k) <==================="); =#
        # Al inicio de la iteración determinamos y obtenemos la muestra de tamaño (batch size)
        # sobre el conjunto de entrenamiento, este lote se utilizará para el cálculo del gradiente
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        yt_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
    
        f           = func(θ, f_params_sample...)
        g           = grad(θ, f_params_sample...)
        llgll       = norm(g)

        # Trust-Region subproblem
        if k == 0 || size(S,2) == 0
            p       = -delta*(g/llgll)
            v_old   =  p
            Bp      =  γ*p
        else
            p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
            # Calculamos factor de ajuste sobre el vector de dirección usando el valor de θ en la iteración previa
            v       = μ * v_old + (θ - θ_old)
            v       = μ * min(1.0, delta/norm(v)) * v
            v_old   = v
            # Actualizamos gradiente direccional
            p       = min(1.0, delta/norm(p + v)) * (p + v)
            Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
        end        

        Q_p         = (p'*(g + 0.5*Bp))[]
        llpll       = norm(p)

        # Actualización del vector de pesos
        θ_new       = θ + p
        f_new       = func(θ_new, f_params_sample...)
        g_new       = grad(θ_new, f_params_sample...)

        # Vectores de curvatura (s,y)
        s           = p
        y           = g_new - g
        
        # Reduction ratio
        ρ           = (f_new - f) / Q_p

        # Condición de aceptación del incremento
        θ_old = θ # Para cálculo de factor de ajuste sobre el gradiente direccional
        f_old = f # Para corrección del tamaño de paso
        θ = θ_new
        f = f_new
        g = g_new
        Θ = hcat(Θ, θ)
        # Condición de paro
        if llgll < tol
            println("Condición de paro")
            return
        end

        # Corrección de tamaño de paso de acuerdo con el incremento obtenido y el coeficiente de reset
        R_k = f - f_old
        if mod(k, K) == 0
            # Obtenemos nuevo batch y evaluamos
            idxs = sample(axes(f_params.Xt, 1), batch_size)
            # sample
            Xt_sample = f_params.Xt[idxs]
            yt_sample = f_params.yt[idxs]
            # parametros de la funcion objetivo
            f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
            f_k = func(θ, f_params_sample...)
            if f_k - f ≥ -γ1 + γ2 * sum(R_k)
                # Ajuste del tamaño de paso y parámetro de ajuste del región de confianza
                nIter = min(2 * nIter, N)
                if nIter == N
                    ζ = 0
                end
            end        
        end

        # Actualización del radio de la región de confianza
        ρ_hat = ζ*T*ρ_hat + ρ
        T     = ζ*T + 1
        ρ_hat = ρ_hat / T
        if ρ_hat < 0.1
            delta = min(delta, norm(s))
        elseif ρ_hat ≥ 0.5 && norm(s) ≥ delta
            delta = 2*delta
        end
        #= Actualización clásica
        if ρ > 0.75
            if norm(p) ≤ 0.8*delta
                delta_new = delta
            else
                delta_new = 2*delta
            end
        elseif (0.1 ≤ ρ && ρ ≤ 0.75)
            delta_new = delta
        else
            delta_new = 0.5*delta
        end
        delta = delta_new
        =#

        # Actualización de S, Y (Algoritmo 3)
        if ρ < τ && k > 0
            if k == lim_m
                # Generamos conjunto de muestras en una región alrededor de la estimación actual de parámetros
                # para la conformación de los vectores de curvatura s, y
                
            elseif k == 0
                # Restart
                S = []
                Y = []
                k = 0
            end
        else
            # Classic S, Y update
            y_Bs        = y - Bp
            if abs((s'*y_Bs)[]) > 1e-8*llpll*norm(y_Bs)
                #println("Curvature pairs")
                if k == 0
                    S = s
                    Y = y
                else
                    S = [S s]
                    Y = [Y y]
                end
                if (size(S,2) > lim_m)
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end

                if size(S,2) == 0
                    println("S is empty!")
                end

                while (size(S, 2) > 0)                
                    SY      = S'*Y
                    SS      = S'*S
                    if size(SY) == ()
                        LDLt = SY
                    else
                        LDLt = tril(SY) + tril(SY,-1)'
                    end
                    eig_val = eigen(LDLt,SS).values
                    λHat_min= minimum(eig_val)
                    if λHat_min > 0
                        γ   = max(0.5*λHat_min, 1e-6)
                    else
                        γ   = min(1.5*λHat_min, -1e-6)
                    end

                    Minv    = (LDLt - γ*SS)
                    Ψ       = Y - γ*S

                    if size(Ψ,2) == rank(Ψ) && rank(Minv) == size(Minv,2)
                        break
                    else
                        #println("Psi is NOT full column rank! or M is not invertible!")
                        S = S[:, 2:end]
                        Y = Y[:, 2:end]
                    end
                end
            end
        end

        k = k + 1;
    end
    return Θ
end