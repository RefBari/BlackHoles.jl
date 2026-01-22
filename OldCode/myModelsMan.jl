#=
    ODE models for orbital mechanics
=#

function NewtonianOrbitModel(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    numer = (1+e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = numer / denom
    ϕ̇ = numer / denom

    return [χ̇, ϕ̇]

end

function RelativisticOrbitModel(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom )
    ϕ̇ = numer / (M*(p^(3/2))*denom)

    return [χ̇, ϕ̇]

end

function AbstractNNOrbitModel(u, model_params, t; NN=nothing, NN_params=nothing, NN_st=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params, NN_st)[1]
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇]

end

function AbstractNROrbitModel(du, u, model_params, t;
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
        nn_schwarzschild = [1,1]
    else
        nn_schwarzschild = NN_adapter(u, NN_params)
    end

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))
        # print("\n\nTrue g_tt:", -f^(-1))
        p = [p_t, p_r, p_θ, p_ϕ]
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

    c1 = Conservative[1] # ṫ
    c2 = Conservative[2] # ṙ
    c3 = Conservative[3] # θ̇
    c4 = Conservative[4] # ϕ̇
    c5 = Conservative[5] # p_t dot
    c6 = Conservative[6] # p_r dot 
    c7 = Conservative[7] # p_θ dot 
    c8 = Conservative[8] # p_ϕ dot 

    r   = x[2]
    pr  = x[6]
    pφ  = x[8]
    eps = 1e-12

    # Lc_schw(r; G=1.0, M=1.0) = r*sqrt(G*M)/sqrt(max(r-3G*M, 1e-12))  # guard r>3M

    # function eccentricity_inst(radius, radial_velocity, angular_velocity, angular_momentum; Lc = Lc_schw, G=1.0, M=1.0, α=40.0, β=10.0)
    #     r = radius
    #     ṙ = radial_velocity
    #     ϕ̇ = angular_velocity
    #     p_ϕ = angular_momentum
        
    #     Lc_val = Lc(r; G=G, M=M)
    #     radial_eccentricity = (ṙ/(r*ϕ̇))^2
    #     angular_eccentricity = ((abs(p_ϕ)/Lc_val)-1)^2
    #     eccentricity = sqrt(α*radial_eccentricity + β*angular_eccentricity)
    #     return eccentricity
    # end

    # function eccentricity_from_angular_momentum(r, pφ)
    #     L = abs(pφ)
    #     L_circ = sqrt(r/(1-3/r))  # Correct formula
    #     e_proxy = 3 * (abs(L - L_circ) / L_circ)
    #     return min(e_proxy, 0.8)
    # end

    # e = eccentricity_inst(x[2], c2, c4, x[8]; Lc = Lc_schw, α=1, β=4)
    # e_second = eccentricity_from_angular_momentum(x[2], x[8])
    # peters_matthews_factor = (1 - e^2)^(3/2) * (1 + 7*e^2/8)
    # peters_matthews_factor_second = (1 - e_second^2)^(3/2) * (1 + 7*e_second^2/8)
    # print("\npeters-matthews factor: ", peters_matthews_factor_second, " eccentricity: ", e, " eccentricity 2: ", e_second)

    # χ    = (pr^2) / (pr^2 + (pφ / max(r, eps))^2 + eps)  
    # M15 = -1 * (190) * (log(1+exp(nn_schwarzschild[1]))) * (1/x[2]^3)
    # M15 = -376 * (1/x[2]^3)
    # M15 = -1 * (1e-2) * (1/x[2]^3)
    M15 = 0
    M25 = 0
    # M35 = 0
    # M45 = log(1+exp(nn_schwarzschild[1])) # p_ϕ

    M35 = 0
    # M35 = -0.001 * (log(1+exp(nn_schwarzschild[2]))) * M15 * (c6) * (c2)^(-1)
    # M35 = -20 * M15 * (c6) * (c2)^(-1)
    
    # M45 = (1+softplus(nn_schwarzschild[1])) * c4 / (x[2])^2
    # M45  = (1+softplus(nn_schwarzschild[1])) * c4 / (x[2])^2
    # ecc_proxy = eccentricity_proxy(u)
    # peters_matthews_factor = (1 - min(ecc_proxy^2, 0.98))^(3/2)  # Clamp to avoid singularity
    # M45 = (softplus(nn_schwarzschild[1])) * 1 / (x[2])^(-7/2)
    # M45 = softplus(nn_schwarzschild[1])
    # M45 = (1e-2) * 1/softplus(nn_schwarzschild[1])
    # M45 = c4/softplus(nn_schwarzschild[1])
    # M45 = ((x[2])^(-7/2)) * (1 + softplus(nn_schwarzschild[1]))
    # M45 = 0.86 * c4/(x[2])^2
    M45 = (1e-5) * x[8] / (softplus(nn_schwarzschild[1]))
    # M45 = 0
    # print("\nNN Output: ", M45, ", Target:", 0.86 * c4/(x[2])^2)
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