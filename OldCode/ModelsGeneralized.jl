#=
    ODE models for orbital mechanics
=#

# Metric you actually integrate (example: your weak-field inverse metric)
gtt(r) = -(1 - 2/r)^(-1)
gpp(r) =  (r^-2) * (1 + 2/r)^(-1)

dgtt(r) = (2/r^2) * (1 - 2/r)^(-2)  # <-- fixed sign
dgpp(r) = -2*r^(-3)*(1 + 2/r)^(-1) + 2*r^(-4)*(1 + 2/r)^(-2)

function circular_pt_L(R)
    A, B  = gtt(R), gpp(R)
    dA,dB = dgtt(R), dgpp(R)
    # second eq: dA*pt^2 + dB*L^2 = 0  ->  L^2 = -(dA/dB)*pt^2
    pt2 = -1 / (A - B*(dA/dB))     # normalization: A*pt^2 + B*L^2 = -1
    @assert pt2 > 0 "No circular solution at this R"
    pt  = -sqrt(pt2)               # future-directed
    L   = sqrt(-(dA/dB)*pt2)
    return pt, L
end

# Eccentric orbit (returns pt, L) for (p,e)
function eccentric_pt_L(p, e)
    rp = p/(1+e)
    ra = p/(1-e)
    A1, B1 = gtt(rp), gpp(rp)
    A2, B2 = gtt(ra), gpp(ra)
    # (A1 - A2) pt^2 + (B1 - B2) L^2 = 0  ->  L^2 = - ((A1-A2)/(B1-B2)) pt^2
    coef = - ( (A1 - A2) / (B1 - B2) )
    # Use the rp equation to solve for pt^2:
    # A1 pt^2 + B1 L^2 = -1  ->  (A1 + B1*coef) pt^2 = -1
    pt2 = -1 / (A1 + B1*coef)
    pt  = sqrt(pt2)
    L   = sqrt(coef * pt2)
    return pt, L, rp
end

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    # Relativistc Forms
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )

    # Newtonian Forms
    # a = p/(1-e^2)
    # E = -(1 - e^2) / (2*p)
    # L = sqrt(p)
  
    return [M, E, L]
  end


function GENERIC(du, u, model_params, t;
                              NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ
        u[3] = p
        u[4] = e

        q is the mass ratio
    =#
    x = @view u[1:8]

    q = model_params[1]
    M=1.0

    if isnothing(NN)
        conservative_corrections = [0, 0]
        dissipative_corrections = 1.0
    else
        nn_outputs = NN(u, NN_params)
        conservative_corrections = nn_outputs.conservative
        # print("\n u:", u)
        # print("\n NN_params:", NN_params)
        # print("\n nn_outputs:", nn_outputs)
        # print("\n conservative_corrections:", conservative_corrections)
        dissipative_corrections = nn_outputs.dissipative[1]
    end

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))
        
        # f_tt_NN_correction = exp(conservative_corrections[1])
        f_rr_NN_correction = 0 * exp(conservative_corrections[1])
        f_ϕϕ_NN_correction = 0 * sigmoid(conservative_corrections[2])

        # print(f_tt_NN_correction)

        p = [p_t, p_r, p_θ, p_ϕ]

        u = 1 / r 
        f = 1 - 2*u

        # g = [
        #     3333333
        # ]

        # Base Metric: Testing Metric (Frankenstein) 
        # g = [
        #         -f^(-1) 0 0 0;
        #         0 f * (1 + f_rr_NN_correction) 0 0;
        #         0 0 0 0;
        #         0 0 0 r^(-2) * (1 + f_ϕϕ_NN_correction)
        #     ]
        
        # Base Metric: Newtonian Weak-field Limit Metric 
        # g = [
        #         -f^(-1) 0 0 0;
        #         0 ((1 + 2/r)^(-1)) * (1 + f_rr_NN_correction) 0 0;
        #         0 0 0 0;
        #         0 0 0 r^(-2) * (1 + 2/r)^(-1) * (1 + f_ϕϕ_NN_correction)
        #     ]

        # Base Metric: Schwarzschild Metric
        g = [
                -f^(-1) 0 0 0;
                0 f 0 0;
                0 0 0 0;
                0 0 0 r^(-2)
            ]

        H_schwarzschild = (1/2) * p' * g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = J * grad_H

    r   = x[2]
    pr  = x[6]
    pφ  = x[8]
    eps = 1e-12

    M15 = 0
    M25 = 0
    M35 = 0
    # M45 = (1e-5) * x[8] / (softplus(dissipative_corrections))
    M45 = 0
    M55 = 0

    # Dissipation corrections to ...
    Dissipation = [
                    0, # t
                    M15, # r
                    0, # θ
                    M25, # ϕ
                    0, # p_t
                    M35, # p_r
                    0,  # p_θ
                    M45] # p_ϕ

    du_dτ = Conservative + Dissipation

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = M55
    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end