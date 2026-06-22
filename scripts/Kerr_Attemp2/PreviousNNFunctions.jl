```
    PreviousNNFunctions.jl
    Contains most of the functions required to run PreviouslySuccessful.jl

```

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

include("DissipationRicci.jl")

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

function plot_waveform!(gp, tsteps, waveforms_bank, start, stop)
    subgrid = CairoMakie.GridLayout(gp)

    axes = []

    for (k, waveform) in enumerate(waveforms_bank)
        true_h₊ = waveform.true_h₊
        true_hₓ = waveform.true_hₓ
        pred_h₊ = waveform.pred_h₊
        pred_hₓ = waveform.pred_hₓ

        stop_k = min(stop, length(true_h₊), length(pred_h₊))

        idx = start:stop_k

        title_string = @sprintf(
            "%s: θ_obs = %.3f, ϕ_obs = %.3f",
            waveform.name,
            waveform.θ_obs,
            waveform.ϕ_obs
        )

        ax = CairoMakie.Axis(
            subgrid[k, 1];
            title = title_string, 
            xlabel = k == length(waveforms_bank) ? L"Time $(M)$" : "",
            ylabel = L"h(t)"
        )

        CairoMakie.lines!(ax, tsteps[idx], pred_h₊[idx];
            linewidth = 2,
            linestyle = :dash,
            label = L"h_+ \; \mathrm{Predicted}",
            color = :red
        )

        CairoMakie.lines!(ax, tsteps[idx], true_h₊[idx];
            linewidth = 2,
            linestyle = :solid,
            label = L"h_+ \; \mathrm{True}",
            color = :red
        )

        CairoMakie.lines!(ax, tsteps[idx], true_hₓ[idx];
            linewidth = 2,
            linestyle = :solid,
            label = L"h_\times \; \mathrm{True}",
            color = :blue
        )

        CairoMakie.lines!(ax, tsteps[idx], pred_hₓ[idx];
            linewidth = 2,
            linestyle = :dash,
            label = L"h_\times \; \mathrm{Predicted}",
            color = :blue
        )

        if k == 1
            CairoMakie.axislegend(ax; position = :rt)
        end
        
        push!(axes, ax)
    end

    return axes
end

function plot_geodesic_3d!(gp, pred_soln, true_soln; M = 1.0, a = a_true)
    N = size(pred_soln, 2)

    r, θ, ϕ       = pred_soln[2,:], pred_soln[3,:], pred_soln[4,:]
    r_t, θ_t, ϕ_t = true_soln[2,1:N], true_soln[3,1:N], true_soln[4,1:N]

    x   = @. sqrt(r^2+a^2)   * sin(θ)   * cos(ϕ);   y   = @. sqrt(r^2+a^2)   * sin(θ)   * sin(ϕ);   z   = @. r   * cos(θ)
    x_t = @. sqrt(r_t^2+a^2) * sin(θ_t) * cos(ϕ_t); y_t = @. sqrt(r_t^2+a^2) * sin(θ_t) * sin(ϕ_t); z_t = @. r_t * cos(θ_t)

    ax = CairoMakie.Axis3(gp, aspect = :data,
                          xlabel = L"x", ylabel = L"y", zlabel = L"z",
                          title  = "Geodesic")

    CairoMakie.lines!(ax, x_t, y_t, z_t; linewidth = 2, label = "True")
    CairoMakie.lines!(ax, x,   y,   z;   linewidth = 2, label = "Predicted", linestyle = :dash)

    rH = M + sqrt(M^2 - a^2)
    θg = range(0, π,  length = 25)
    ϕg = range(0, 2π, length = 50)
    X = [rH*sin(th)*cos(ph) for th in θg, ph in ϕg]
    Y = [rH*sin(th)*sin(ph) for th in θg, ph in ϕg]
    Z = [rH*cos(th)         for th in θg, ph in ϕg]
    CairoMakie.surface!(ax, X, Y, Z; alpha = 0.15)

    return ax
end

function plot_final_biases!(gp, p)
    gl = CairoMakie.GridLayout(gp)
    
    biases = p.conservative.layer_3.bias
    n_outputs = length(biases)
    
    # Reshape to column vector for heatmap
    bias_matrix = reshape(biases, (n_outputs, 1))
    
    ax = CairoMakie.Axis(gl[1, 1];
        title = "Final Layer Biases",
        xlabel = "",
        ylabel = "Output Index",
        yticks = (1:n_outputs, ["Out $i" for i in 1:n_outputs]),
        xticks = ([0.5], ["Bias"])
    )
    
    # Heatmap
    max_abs = maximum(abs.(biases))
    hm = CairoMakie.heatmap!(ax, [0.5], 1:n_outputs, bias_matrix;
        colormap = :RdBu,
        colorrange = (-max_abs, max_abs))
    
    # Add text labels with actual values
    for i in 1:n_outputs
        CairoMakie.text!(ax, 0.5, i;
            text = @sprintf("%.3e", biases[i]),
            align = (:center, :center),
            fontsize = 14,
            color = abs(biases[i]) > 0.5*max_abs ? :white : :black)
    end
    
    Colorbar(gl[2, 1], hm;
             label = "Bias Value",
             vertical = false,
             flipaxis = false,
             tellwidth = false)
    
    return ax
end

function plot_orbits_waveforms()
    orbit = Plots.plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
    Plots.plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
    
    h₊_waveform = Plots.plot(waveform_real_ecc, label = "Real", linewidth=2)
    Plots.plot!(waveform_nn_real, label = "Prediction", linewidth=2)
    
    hₓ_waveform = Plots.plot(waveform_imag_ecc, label = "Real", linewidth=2)
    Plots.plot!(waveform_nn_imag, label = "Prediction", linewidth=2)
    
    return orbit, h₊_waveform, hₓ_waveform
end

function NN_adapter(r_max, u, params)
    scale_factor = r_max

    # Conservative network
    conservative_features = [u[2] / scale_factor, (cos(u[3]))^2]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)

    # Dissipative network
    dissipative_features = [u[2] / scale_factor]
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end


function initialize_Conservative_NN(final_layer_weights, final_layer_bias, 
                                     hidden_layer_weights, hidden_layer_bias)
    for (i, layer) in enumerate(NN_Conservative_params)
        if !isempty(layer)
            if i == length(NN_Conservative_params)
                layer.weight .= final_layer_weights * randn(rng, eltype(layer.weight), 
                                                            size(layer.weight))
                if final_layer_bias isa AbstractVector
                    layer.bias .= final_layer_bias
                else
                    layer.bias .= final_layer_bias
                end
            else
                layer.weight .= hidden_layer_weights * randn(rng, eltype(layer.weight), 
                                                              size(layer.weight))
                layer.bias   .= hidden_layer_bias * randn(rng, eltype(layer.bias), 
                                                           size(layer.bias))
            end
        end
    end
end

function initialize_Dissipative_NN(final_layer_weights, final_layer_bias, hidden_layer_weights, hidden_layer_bias)
    for (i, layer) in enumerate(NN_Dissipative_params)
        if !isempty(layer)
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

function plot_metric_components!(gp, parameters, r_min, r_max, a_true)
    gl = CairoMakie.GridLayout(gp)
    M, a = 1.0, a_true

    # Grid over the orbital domain
    r_vals  = range(r_min, r_max * 2, length = 40)
    c2_vals = range(0.0, 0.99, length = 40)   # cos²θ ∈ [0, 1]

    # Evaluate true Kerr and NN on the 2D grid
    g_tt_Kerr = [begin
                     θ  = acos(sqrt(c2))
                     sv = @SVector [0.0, r, θ, 0.0]
                     g_Kerr(sv, M=M, a=a)[1,1]
                 end for r in r_vals, c2 in c2_vals]

    g_rr_Kerr = [begin
                     θ  = acos(sqrt(c2))
                     sv = @SVector [0.0, r, θ, 0.0]
                     g_Kerr(sv, M=M, a=a)[2,2]
                 end for r in r_vals, c2 in c2_vals]

    g_θθ_Kerr = [begin
                     θ  = acos(sqrt(c2))
                     sv = @SVector [0.0, r, θ, 0.0]
                     g_Kerr(sv, M=M, a=a)[3,3]
                 end for r in r_vals, c2 in c2_vals]

    g_ϕϕ_Kerr = [begin
                 θ  = acos(sqrt(c2))
                 sv = @SVector [0.0, r, θ, 0.0]
                 g_Kerr(sv, M=M, a=a)[4,4]
             end for r in r_vals, c2 in c2_vals]

    g_tϕ_Kerr = [begin
                 θ  = acos(sqrt(c2))
                 sv = @SVector [0.0, r, θ, 0.0]
                 g_Kerr(sv, M=M, a=a)[1,4]
             end for r in r_vals, c2 in c2_vals]

    g_tt_NN = [begin
               θ   = acos(sqrt(c2))
               sv  = @SVector [0.0, r, θ, 0.0]
               out, _ = NN_Conservative([r / r_max, c2],   # cos²θ, sin²θ
                                        parameters.conservative,
                                        NN_Conservative_state)
               g_NeuralNetwork(sv, out)[1,1]
           end for r in r_vals, c2 in c2_vals]   # ← only 2 ranges, produces 2D matrix
    
    g_rr_NN = [begin
                   θ   = acos(sqrt(c2))
                   sv  = @SVector [0.0, r, θ, 0.0]
                   out, _ = NN_Conservative([r / r_max, c2],   # cos(θ) = sqrt(cos²θ)
                                            parameters.conservative,
                                            NN_Conservative_state)
                   g_NeuralNetwork(sv, out)[2,2]
               end for r in r_vals, c2 in c2_vals]

    g_θθ_NN = [begin
                   θ   = acos(sqrt(c2))
                   sv  = @SVector [0.0, r, θ, 0.0]
                   out, _ = NN_Conservative([r / r_max, c2],   # cos(θ) = sqrt(cos²θ)
                                            parameters.conservative,
                                            NN_Conservative_state)
                   g_NeuralNetwork(sv, out)[3,3]
               end for r in r_vals, c2 in c2_vals]

    g_ϕϕ_NN = [begin
                   θ   = acos(sqrt(c2))
                   sv  = @SVector [0.0, r, θ, 0.0]
                   out, _ = NN_Conservative([r / r_max, c2],   # cos(θ) = sqrt(cos²θ)
                                            parameters.conservative,
                                            NN_Conservative_state)
                   g_NeuralNetwork(sv, out)[4,4]
               end for r in r_vals, c2 in c2_vals]

    g_tϕ_NN = [begin
                   θ   = acos(sqrt(c2))
                   sv  = @SVector [0.0, r, θ, 0.0]
                   out, _ = NN_Conservative([r / r_max, c2],   # cos(θ) = sqrt(cos²θ)
                                            parameters.conservative,
                                            NN_Conservative_state)
                   g_NeuralNetwork(sv, out)[1,4]
               end for r in r_vals, c2 in c2_vals]

    r_grid  = collect(r_vals)
    c2_grid = collect(c2_vals)

    # ---- 3-panel layout inside gp ----
    ax1 = CairoMakie.Axis3(gl[1, 1],
        title    = L"Kerr $g_{tt}(r, \cos^2\theta)$",
        xlabel   = L"r",
        ylabel   = L"\cos^2\theta",
        zlabel   = L"g_{tt}",
        azimuth  = 1.3π,
        elevation = 0.2π)

    ax2 = CairoMakie.Axis3(gl[1, 2],
        title    = L"Kerr $\hat{g}_{rr}(r, \cos^2\theta)$",
        xlabel   = L"r",
        ylabel   = L"\cos^2\theta",
        zlabel   = L"g_{rr}",
        azimuth  = 1.3π,
        elevation = 0.2π)

    ax3 = CairoMakie.Axis3(gl[1, 3],
        title    = L"Kerr $\hat{g}_{\theta\theta}(r, \cos^2\theta)$",
        xlabel   = L"r",
        ylabel   = L"\cos^2\theta",
        zlabel   = L"g_{\theta\theta}",
        azimuth  = 1.3π,
        elevation = 0.2π)

    ax4 = CairoMakie.Axis3(gl[3, 1],
        title    = L"Kerr $\hat{g}_{\phi\phi}(r, \cos^2\theta)$",
        xlabel   = L"r",
        ylabel   = L"\cos^2\theta",
        zlabel   = L"g_{\phi\phi}",
        azimuth  = 1.3π,
        elevation = 0.2π)

    ax5 = CairoMakie.Axis3(gl[3, 2],
        title    = L"Kerr $\hat{g}_{t\phi}(r, \cos^2\theta)$",
        xlabel   = L"r",
        ylabel   = L"\cos^2\theta",
        zlabel   = L"g_{t\phi}",
        azimuth  = 1.3π,
        elevation = 0.2π)

    # Shared colormap for Kerr and NN so they're visually comparable
    clims_tt = (min(minimum(g_tt_Kerr), minimum(g_tt_NN)),
                max(maximum(g_tt_Kerr), maximum(g_tt_NN)))
    clims_rr = (min(minimum(g_rr_Kerr), minimum(g_rr_NN)),
                max(maximum(g_rr_Kerr), maximum(g_rr_NN)))
    clims_θθ = (min(minimum(g_θθ_Kerr), minimum(g_θθ_NN)),
                max(maximum(g_θθ_Kerr), maximum(g_θθ_NN)))
    clims_ϕϕ = (min(minimum(g_ϕϕ_Kerr), minimum(g_ϕϕ_NN)),
                max(maximum(g_ϕϕ_Kerr), maximum(g_ϕϕ_NN)))
    clims_tϕ = (min(minimum(g_tϕ_Kerr), minimum(g_tϕ_NN)),
                max(maximum(g_tϕ_Kerr), maximum(g_tϕ_NN)))

    sf1 = CairoMakie.surface!(ax1, r_grid, c2_grid, g_tt_Kerr;
                               colormap  = :heat, colorrange = clims_tt, label = L"True g_{tt}", alpha = 0.85)
    CairoMakie.surface!(ax1, r_grid, c2_grid, g_tt_NN;
                               colormap = :viridis, colorrange = clims_tt, label = L"NN g_{tt}", alpha = 0.55)
    CairoMakie.wireframe!(ax1, r_grid, c2_grid, g_tt_NN;
                                color = :black, linewidth = 0.4, alpha = 0.3)

    sf3 = CairoMakie.surface!(ax2, r_grid, c2_grid, g_rr_Kerr;
                               colormap = :heat, colorrange = clims_rr, label = L"True g_{rr}", alpha = 0.85)
    CairoMakie.surface!(ax2, r_grid, c2_grid, g_rr_NN;
                               colormap = :viridis, colorrange = clims_rr, label = L"NN g_{rr}", alpha = 0.55)
    CairoMakie.wireframe!(ax2, r_grid, c2_grid, g_rr_NN;
                                color = :black, linewidth = 0.4, alpha = 0.3)

    sf5 = CairoMakie.surface!(ax3, r_grid, c2_grid, g_θθ_Kerr;
                               colormap = :heat, colorrange = clims_θθ, label = L"True g_{\theta\theta}", alpha = 0.85)
    CairoMakie.surface!(ax3, r_grid, c2_grid, g_θθ_NN;
                               colormap = :viridis, colorrange = clims_θθ, label = L"NN g_{\theta\theta}", alpha = 0.55)
    CairoMakie.wireframe!(ax3, r_grid, c2_grid, g_θθ_NN;
                                color = :black, linewidth = 0.4, alpha = 0.3)

    sf7 = CairoMakie.surface!(ax4, r_grid, c2_grid, g_ϕϕ_Kerr;
                               colormap = :heat, colorrange = clims_ϕϕ, label = L"True g_{\phi\phi}", alpha = 0.85)
    CairoMakie.surface!(ax4, r_grid, c2_grid, g_ϕϕ_NN;
                               colormap = :viridis, colorrange = clims_ϕϕ, label = L"NN g_{\phi\phi}", alpha = 0.55)
    CairoMakie.wireframe!(ax4, r_grid, c2_grid, g_ϕϕ_NN;
                                color = :black, linewidth = 0.4, alpha = 0.3)

    sf9 = CairoMakie.surface!(ax5, r_grid, c2_grid, g_tϕ_Kerr;
                               colormap = :heat, colorrange = clims_tϕ, label = L"True g_{t\phi}", alpha = 0.85)
    CairoMakie.surface!(ax5, r_grid, c2_grid, g_tϕ_NN;
                               colormap = :viridis, colorrange = clims_tϕ, label = L"NN g_{t\phi}", alpha = 0.55)
    CairoMakie.wireframe!(ax5, r_grid, c2_grid, g_tϕ_NN;
                                color = :black, linewidth = 0.4, alpha = 0.3)

    Colorbar(gl[2, 1], sf1;
             label       = L"g_{tt}",
             vertical    = false,
             flipaxis    = false,
             tellwidth   = false)

    Colorbar(gl[2, 2], sf3;
             label       = L"g_{rr}",
             vertical    = false,
             flipaxis    = false,
             tellwidth   = false)

    Colorbar(gl[2, 3], sf5;
             label       = L"g_{\theta\theta}",
             vertical    = false,
             flipaxis    = false,
             tellwidth   = false)

    Colorbar(gl[4, 1], sf7;
             label       = L"g_{\phi\phi}",
             vertical    = false,
             flipaxis    = false,
             tellwidth   = false)

    Colorbar(gl[4, 2], sf9;
             label       = L"g_{t\phi}",
             vertical    = false,
             flipaxis    = false,
             tellwidth   = false)

    return ax1, ax2
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

function Newtonian(r_max, du, u, model_params, t;
                              NN=NN_adapter, NN_params=nothing)
    x = SVector{8}(u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8])

    q = model_params[1]
    M = 1.0

    nn_outputs = NN_adapter(r_max, u, NN_params)
    dissipation_raw = nn_outputs.dissipative
   
    # Φ_flux = 1e-2 * softplus(dissipation_raw[1])
    Φ_flux = 0

    function H(state_vec; a = 0.9)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        nn_outputs = NN_adapter(r_max, state_vec, NN_params)
        conservative_corrections = nn_outputs.conservative
        r_val = state_vec[2]
        θ_val = state_vec[3]
    
        g = g_NeuralNetwork(state_vec, conservative_corrections)
        inv_g = inv(g)

        p = SVector(p_t, p_r, p_θ, p_ϕ)
        H_Kerr = (1/2) * p' * inv_g * p

        return H_Kerr # Returns equations of motion in PROPER time
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

    du[1] = 1

    return du
end

function solve_kerr(u0; M=1.0, a=0.9, λmax=1e3, saveat=0.1,
                           reltol=1e-10, abstol=1e-10)

    kerr_ode! = (du, u, p, t) -> Kerr(du, u, [1.0], t; M=p[1], a=p[2])
    prob = ODEProblem(kerr_ode!, u0, (0.0, λmax), (M, a))
    sol  = solve(prob, Vern9(); reltol=reltol, abstol=abstol, saveat=saveat)

    return sol
end