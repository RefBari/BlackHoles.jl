using LaTeXStrings
using Measures
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
using ComponentArrays
using Optimization
using OptimizationOptimisers
using Printf
using Lux
using ProgressMeter
using Zygote
using NNlib: softplus

function get_weights_biases(params)
    conservative = params.conservative

    # LAYER 1
    weight_1 = conservative.layer_1.weight
    bias_1 = conservative.layer_1.bias

    # LAYER 2
    weight_2 = conservative.layer_2.weight
    bias_2 = conservative.layer_2.bias

    # LAYER 3
    weight_3 = conservative.layer_3.weight
    bias_3 = conservative.layer_3.bias

    return (weight_1, bias_1, 
            weight_2, bias_2,
            weight_3, bias_3)
end

function plot_waveform(tsteps, waveform_real, pred_waveform, start, stop)
    waveforms = plot(tsteps[start:stop], waveform_real[start:stop], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True",
                    xaxis = "Time", ylabel = L"h_{+}(t)", title =  L"h_{+}(t)")
    plot!(waveforms, tsteps[start:stop], pred_waveform[start:stop], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "Predicted")
    return waveforms
end

function plot_orbits_waveforms()
    orbit = plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
    plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
    
    h₊_waveform = plot(waveform_real_ecc, label = "Real", linewidth=2)
    plot!(waveform_nn_real, label = "Prediction", linewidth=2)
    
    hₓ_waveform = plot(waveform_imag_ecc, label = "Real", linewidth=2)
    plot!(waveform_nn_imag, label = "Prediction", linewidth=2)
    
    return orbit, h₊_waveform, hₓ_waveform
end

function NN_adapter(u, params)
    scale_factor = r_max

    # Conservative network
    conservative_features = [u[2] / scale_factor]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2] / scale_factor]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end


function initialize_Conservative_NN(final_layer_weights, final_layer_bias, hidden_layer_weights, hidden_layer_bias)
    for (i, layer) in enumerate(NN_Conservative_params)
        if ~isempty(layer)
            if i == length(NN_Conservative_params)
                layer.weight .= final_layer_weights * randn(rng, eltype(layer.weight), size(layer.weight))
                layer.bias .= final_layer_bias
            else 
                layer.weight .= hidden_layer_weights * randn(rng, eltype(layer.weight), size(layer.weight))
                layer.bias .= hidden_layer_bias * randn(rng, eltype(layer.bias), size(layer.bias))
            end
        end
    end
end

function initialize_Dissipative_NN(final_layer_weights, final_layer_bias, hidden_layer_weights, hidden_layer_bias)
    for (i, layer) in enumerate(NN_Dissipative_params)
        if ~isempty(layer)
            if i == length(NN_Dissipative_params)
                layer.weight .= final_layer_weights * randn(rng, eltype(layer.weight), size(layer.weight))
                layer.bias .= final_layer_bias
            else 
                layer.weight .= hidden_layer_weights * randn(rng, eltype(layer.weight), size(layer.weight))
                layer.bias .= hidden_layer_bias * randn(rng, eltype(layer.bias), size(layer.bias))
            end
        end
    end
end

function plot_metric_components(parameters, r_min, r_max)
    r_values = range(r_min, r_max, length = 30)

    f_tt_predicted = []
    f_rr_predicted = []

    Φ_predicted = []
    scale_factor = r_max
    
    for r in r_values
        output, _ = NN_Conservative([r / scale_factor], parameters.conservative, NN_Conservative_state)
        push!(f_tt_predicted, 1 - softplus(output[1]))
        push!(f_rr_predicted, 1 + softplus(output[2]))

        output, _ = NN_Dissipative([r / scale_factor], parameters.dissipative, NN_Dissipative_state)
        push!(Φ_predicted, 5e-2 * softplus(output[1]))
    end

    g_tt_Schwarzschild = [-(1 .- 2 ./ r) for (r, f) in zip(r_values, f_tt_predicted)]
    g_rr_Schwarzschild = [(1 .- 2 ./ r)^(-1) for (r, f) in zip(r_values, f_rr_predicted)]

    g_tt_NN = [-1 * f_tt for (r, f_tt) in zip(r_values, f_tt_predicted)]
    g_rr_NN = [1 * f_rr for (r, f_rr) in zip(r_values, f_rr_predicted)]

    g_tt_Minkowski = [-1 for (r, f_tt) in zip(r_values, f_tt_predicted)]
    g_rr_Minkowski = [1 for (r, f_rr) in zip(r_values, f_rr_predicted)]

    predicted_flux_plots = plot(r_values, -1 * Φ_predicted, lw=2, color=:green, xlabel=L"r", ylabel=L"\Phi(r)", title="Dissipation", label = "Predicted")
    plot!(predicted_flux_plots, r_values, [-3.45e-1 * (r)^(-7/2) for r in r_values], lw=2, ls=:dash, color=:blue, label="True", legend=:topright)

    g_tt_plots = plot(r_values, g_tt_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_tt_plots, r_values, g_tt_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")
    plot!(g_tt_plots, r_values, g_tt_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")

    g_rr_plots = plot(r_values, g_rr_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_rr_plots, r_values, g_rr_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")
    plot!(g_rr_plots, r_values, g_rr_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")

    return g_tt_plots, g_rr_plots, predicted_flux_plots
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

    if isnothing(NN)
        C1 = 0
    else
        nn_outputs = NN(u, NN_params)
        dissipation_raw = nn_outputs.dissipative
    end

    Φ_flux = 5e-2 * softplus(dissipation_raw[1])

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        if isnothing(NN)
            conservative_corrections = 0.0
        else
            nn_outputs = NN(state_vec, NN_params)
            conservative_corrections = nn_outputs.conservative

            raw_tt = conservative_corrections[1]
            raw_rr = conservative_corrections[2]

            f_tt_NN_correction = 1 - softplus(raw_tt)
            f_rr_NN_correction = 1 + softplus(raw_rr)
        end

        p = SVector(p_t, p_r, 0, p_ϕ)

        # Base Metric: Minkowski Metric
        g = @SMatrix [
                -1*(f_tt_NN_correction) 0 0 0;
                0 1*(f_rr_NN_correction) 0 0;
                0 0 r^2 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
            
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    Conservative = J * grad_H

    Dissipative = [0, 0, 0, 0, 0, 0, 0, -Φ_flux]

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