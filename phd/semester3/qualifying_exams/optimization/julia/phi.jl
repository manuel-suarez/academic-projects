function phi(σ, Δ, a, λ)
    obs_tol = 1e-10
    t = λ .+ σ
    if (sum(abs.(a) .< obs_tol) > 0) || (sum(abs.(t) .< obs_tol) > 0)
        llpll2 = 0
        for i in 1:length(a)
            if (abs(a[i]) > obs_tol) && (abs(t[i]) > obs_tol)
                return -1/Δ
            elseif (abs(a[i]) > obs_tol) && (abs(t[i]) > obs_tol)
                llpll2 = llpll2 + (a[i]/t[i])^2
            end
        end
        return 1/sqrt(llpll2) - 1/Δ
    end
    llpll = norm(a./t)
    phi = 1/llpll - 1/Δ
end