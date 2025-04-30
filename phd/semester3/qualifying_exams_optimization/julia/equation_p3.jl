function equation_p3(λ_min, Δ, p_hat, λ, P_ll)
    eq_tol = 1e-10
    α = sqrt(Δ^2 - p_hat'*p_hat)

    # case 1:lambda[1]
    if (abs(λ_min - λ[1]) < eq_tol)
        u_min   = P_ll[:,1]/norm(P_ll[:,1])
        z_star  = α * u_min
    # case 2:gamma
    else
        n,k = size(P_ll)
        e   = zeros(n,1)
        found = 0
        for i in 1:k
            e[i] = 1
            u_min = e - P_ll*P_ll[i,1:end]'
            if (norm(u_min) > eq_tol)
                found = 1
                break
            end
            e[i] = 0
        end
        if found == 0
            e[i+1] = 1
            u_min = e - P_ll*P_ll[i+1,:]'
        end
        u_min = u_min/norm(u_min)
        z_star = α * u_min
    end
    p_star = p_hat + z_star
    return p_star
end