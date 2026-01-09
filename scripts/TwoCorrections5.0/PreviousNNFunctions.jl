include("TrainingDataFunctions.jl")


function plot_metric_components(parameters, r_min, r_max)
    r_values = range(r_min, r_max, length = 30)

    f_tt_predicted = []
    f_rr_predicted = []

    semilatus_rectum =  2 * (r_min * r_max) / (r_min + r_max)
    scale_factor = semilatus_rectum
    
    for r in r_values
        output, _ = NN_Conservative([r / scale_factor], parameters.conservative, NN_Conservative_state)
        push!(f_tt_predicted, 1 - softplus(output[1]))
        push!(f_rr_predicted, 1/(1 - softplus(output[1])))
    end

    g_tt_Schwarzschild = [-(1 .- 2 ./ r) for (r, f) in zip(r_values, f_tt_predicted)]
    g_rr_Schwarzschild = [(1 .- 2 ./ r)^(-1) for (r, f) in zip(r_values, f_rr_predicted)]

    g_tt_NN = [-1 * f_tt for (r, f_tt) in zip(r_values, f_tt_predicted)]
    g_rr_NN = [1 * f_rr for (r, f_rr) in zip(r_values, f_rr_predicted)]

    g_tt_Minkowski = [-1 for (r, f_tt) in zip(r_values, f_tt_predicted)]
    g_rr_Minkowski = [1 for (r, f_rr) in zip(r_values, f_rr_predicted)]

    g_tt_plots = plot(r_values, g_tt_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_tt_plots, r_values, g_tt_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")
    plot!(g_tt_plots, r_values, g_tt_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")

    g_rr_plots = plot(r_values, g_rr_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_rr_plots, r_values, g_rr_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")
    plot!(g_rr_plots, r_values, g_rr_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")

    return g_tt_plots, g_rr_plots
end

function NewtonianIC(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity

    # Newtonian Forms
    E = -(1 - e^2) / (2*p)
    L = sqrt(p)
  
    return [M, E, L]
  end

const J = SMatrix{8,8}([
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ])

function Newtonian(du, u, model_params, t;
                              NN=nothing, NN_params=nothing)
    x = SVector{8}(u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8])

    q = model_params[1]
    M = 1.0

    dissipation_raw = 0.0
    if isnothing(NN)
        C1, C2, C3, C4 = 0.0, 0.0, 0.0, 0.0
    else
        nn_outputs = NN(u, NN_params)
        dissipation_raw = nn_outputs.dissipative 
    end

    f_ϕ_flux = 1e-2 * softplus(dissipation_raw[1])

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        if isnothing(NN)
            conservative_corrections = 0.0
        else
            nn_outputs = NN(state_vec, NN_params)
            conservative_corrections = nn_outputs.conservative
        end

        f_rr_NN_correction = 1 - softplus(conservative_corrections[1])

        p = SVector(p_t, p_r, 0, p_ϕ)

        # Base Metric: Minkowski Metric
        g = @SMatrix [
                -1*(f_rr_NN_correction) 0 0 0;
                0 1*(f_rr_NN_correction)^(-1) 0 0;
                0 0 1 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
            
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    Conservative = J * grad_H

    Dissipative = [0, 0, 0, 0, 0, 0, 0, -f_ϕ_flux]

    du_dτ = Conservative + Dissipative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = 0
    du[1] = 1

    return du
end