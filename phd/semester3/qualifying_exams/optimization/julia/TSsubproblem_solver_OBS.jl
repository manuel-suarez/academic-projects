include("phi.jl")
include("equation_p1.jl")
include("equation_p2.jl")
include("equation_p3.jl")
include("newton_method.jl")

function TRsubproblem_solver_OBS(Δ, γ, g, Ψ, Minv)
    obs_tol             = 1e-10    
    # Factorización QR
    Q, R                = qr(Ψ)
    Q                   = Q[:,1:size(R,2)] # Extraemos las columnas requeridas
    RMR                 = R*(Minv\R')
    RMR                 = (RMR + RMR')./2
    # Eingenvalores
    eig_va, eig_ve      = eigen(RMR)  # Los eigenvalores ya se encuentran en un vector por lo que no requerimos obtenerlos con diag
    λ_hat               = sort(eig_va)
    idx                 = sortperm(eig_va)
    U                   = eig_ve[:,idx]
    λ1                  = λ_hat .+ γ
    λ                   = [λ1; γ]
    # Min eigenvalor
    λ                   = λ.*(abs.(λ) .> obs_tol)
    λ_min               = min(λ[1], γ)
    # P_LL, |p_perb|, a
    P_ll                = Q*U
    g_ll                = P_ll'*g
    gTg                 = (g'*g)[]
    g_llTg_ll           = (g_ll'*g_ll)[]
    llg_perbll          = sqrt(abs(gTg - g_llTg_ll))

    if llg_perbll^2 < obs_tol
        llg_perbll = 0
    end
    a                   = [g_ll; llg_perbll]
    # 3 cases + Newton
    # case 1
    if (λ_min > 0) && (phi(0, Δ, a, λ) ≥ 0)
        σ_star          = 0
        τ_star          = γ + σ_star
        p_star          = equation_p1(τ_star, g, Ψ, Minv)
    # case 2
    elseif (λ_min ≤ 0) && (phi(-λ_min, Δ, a, λ) >= 0)
        σ_star          = -λ_min
        p_star          = equation_p2(σ_star, γ, g, a, λ, P_ll, g_ll)
    # case 3
        if (λ_min < 0) # Hard case
            p_hat = p_star
            p_star = equation_p3(λ_min, Δ, p_hat, λ, P_ll)
        end
    # newton
    else
        if λ_min > 0
            σ_star = newton_method(0, Δ, a, λ)
        else
            σ_hat = maximum(abs.(a)./Δ - λ)
            if σ_hat > -λ_min
                σ_star = newton_method(σ_hat, Δ, a, λ)
            else
                σ_star = newton_method(-λ_min, Δ, a, λ)
            end
        end
        τ_star = σ_star + γ
        p_star = equation_p1(τ_star, g, Ψ, Minv)
    end
    return p_star
end

