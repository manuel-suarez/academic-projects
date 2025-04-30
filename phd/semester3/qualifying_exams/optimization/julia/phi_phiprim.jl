function phi_phiprim(σ, Δ, a, λ)
    obs_tol = 1e-10
    t = λ .+ σ
    if (sum(abs.(a) .< obs_tol) > 0) || (sum(abs.(t) .< obs_tol) > 0)
        llpll2 = 0
        llpll_prim = 0
        for i in 1:length(a)
            if (abs(a[i]) > obs_tol) && (abs(t[i]) > obs_tol)
                return -1/Δ, 1/obs_tol
            elseif (abs(a[i]) > obs_tol) && (abs(t[i]) > obs_tol)
                llpll2 = llpll2 + (a[i]/t[i])^2
                llpll_prim = llpll_prim + (a[i]^2/t[i]^3)
            end
        end
        llpll = sqrt(llpll2)
        ϕ = 1/llpll - 1/Δ
        ϕ_prim = llpll_prim / llpll^3
        return ϕ, ϕ_prim
    end
    llpll = norm(a./t)
    ϕ = 1/llpll - 1/Δ
    ϕ_prim = sum(a.^2 ./ t.^3) / llpll^3
    return ϕ, ϕ_prim
end