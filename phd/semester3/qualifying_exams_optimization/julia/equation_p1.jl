function equation_p1(τ, g, Ψ, Minv)
    Z = τ * Minv + Ψ'*Ψ
    f = Ψ'*g
    p = -(g - Ψ*(Z\f)) / τ
    return p
end