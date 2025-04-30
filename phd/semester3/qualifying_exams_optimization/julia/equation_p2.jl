function equation_p2(σ, γ, g, a, λ, P_ll, g_ll)
    eq_tol = 1e-10
    t           = λ .+ σ
    idx         = findall(abs.(t) .> eq_tol)
    c           = length(t)
    v           = zeros(c,1)
    v[idx]      = a[idx]./(λ[idx] .+ σ)
    if abs(γ + σ) < eq_tol
        p = -P_ll*v[1:c-1]
    else
        p = -P_ll*v[1:c-1] - (g - P_ll*g_ll)/(γ+σ)
    end
    return p
end
