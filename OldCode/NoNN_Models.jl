#=
    ODE models for orbital mechanics
=#

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )
  
    return [M, E, L]
  end


function GENERIC(du, u, model_params, t;
                              NN=nothing, NN_params=nothing)
    x = @view u[1:8]
    q = model_params[1]
    M=1.0

    if isnothing(NN)
        conservative_corrections = [0, 0, 0]
        dissipative_corrections = 1.0
    else
        nn_outputs = NN(u, NN_params)
        conservative_corrections = nn_outputs.conservative
        dissipative_corrections = nn_outputs.dissipative[1]
    end

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))
        
        f_tt_NN_correction = exp(conservative_corrections[1])
        f_rr_NN_correction = exp(conservative_corrections[2])
        f_ϕϕ_NN_correction = exp(conservative_corrections[3])

        p = [p_t, p_r, p_θ, p_ϕ]
        g = [
                -f^(-1) * f_tt_NN_correction 0 0 0;
                0 f * f_rr_NN_correction 0 0;
                0 0 0 0;
                0 0 0 r^(-2) * f_ϕϕ_NN_correction
            ]

        H_schwarzschild = (1/2) * p' * g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    L = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = L * grad_H

    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    M55 = 0
    du[9] = M55
    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end