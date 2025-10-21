using QuadGK

function asset_pricing_baseline(t, c, g, r, x_0; T_max=2000.0)
    x(s) = (x_0 + c / g) * exp(g * s) - c / g
    discount_x(s) = exp(-r * s) * x(s)
    return [quadgk(discount_x, t_val, T_max)[1] for t_val in t] .* exp.(r .* t)
end
