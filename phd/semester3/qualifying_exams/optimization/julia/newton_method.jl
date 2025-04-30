include("phi_phiprim.jl")
function newton_method(σ_0, Δ, a, λ)
    newton_tol      = 1e-10
    newton_maxit    = 100

    k               = 0
    σ               = σ_0
    ϕ, ϕ_prim       = phi_phiprim(σ, Δ, a, λ)
    while (abs(ϕ)>newton_tol) && (k < newton_maxit)
        σ = σ - ϕ/ϕ_prim
        ϕ, ϕ_prim = phi_phiprim(σ, Δ, a, λ)
        k = k + 1
    end
    return σ
end