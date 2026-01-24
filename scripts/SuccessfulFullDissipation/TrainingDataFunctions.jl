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

include("2021EquationFunctions.jl")
include("Orbit2Waveform.jl")

semilatusRectum = 10
eccentricity = 0.4

w1 = [-0.39837056; 1.4048915; -0.90800977; -0.46240625; -0.7634825; 0.74073714; 0.43308562; 0.06071772; -1.7889463; -0.56506187;;]
b1 = [-1.544076, 0.1036469, -1.10246, 0.11249606, -0.7712159, 1.2483029, -0.9609448, 0.505475, -0.12751098, 1.1966323]

w2 = [-0.44270977 -0.2882591 -0.8021883 -2.7287295 0.8085699 -1.0365628 0.1537832 -0.3124268 -0.73789054 -0.2400705; -1.0578264 0.5180804 -1.2093686 0.43017146 0.6965366 0.3625032 0.050369717 0.5568215 0.42286134 0.17354445; 2.2194676 -1.4117541 -1.1716135 0.37777776 -2.5717888 2.2017038 1.6425238 1.9930266 -1.981764 -0.9884478; 0.1449852 1.326198 -0.8658664 0.16984919 -0.16200748 -1.3098066 -0.81082004 -0.88312525 -0.5778299 -0.49465936; -0.1766802 -0.82277626 0.87170935 -1.4021137 -1.5870486 0.4365277 1.3888034 0.5956235 0.13407159 -0.0018043438; 2.1651797 -0.60533977 -0.05052091 -0.9646062 0.2832133 -0.49201268 0.40630192 0.9413386 -0.2543453 -0.077779315; -1.2130032 0.021851407 1.6846828 1.7499614 0.93378615 -1.4633826 1.1086296 0.38150528 -0.0055636438 -0.93811333; -0.41744497 2.1837971 1.2539394 -0.9891464 0.9938346 0.83762413 0.4205669 -0.8964099 -0.85912293 0.57681113; -0.9282118 0.25228012 -1.984959 -0.66282004 -1.2207322 0.063692614 -0.12482774 0.33321264 -0.080280215 -0.4872967; -0.8598793 0.7317451 0.21588017 -1.0917567 1.2147632 -0.4028765 -1.269087 1.8830181 0.33889103 -0.077292845]
b2 = [-0.37791815, 0.15943322, -0.1305767, 2.9726534, 0.25721514, 0.14376616, -1.6265668, -0.7205314, -1.0319502, 0.52534145]

w3 = [-0.63720554 -0.025167208 0.13269846 0.1768508 -0.08783083 -0.25030404 -0.14630164 -0.8243371 0.09880934 0.8300059; 0.22333677 -0.28092703 -0.277219 -0.2732239 0.22889 0.2779033 0.27756622 -0.18056852 -0.27385405 -0.25700864]
b3 = [-2.8377137, -3.2803266]

rng = MersenneTwister(0)

NN_Conservative = Chain(
    Dense(1, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 2),
)

NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)

NN_Conservative_params.layer_1.weight .= reshape(w1, 10, 1)
NN_Conservative_params.layer_1.bias   .= b1
NN_Conservative_params.layer_2.weight .= w2
NN_Conservative_params.layer_2.bias   .= b2
NN_Conservative_params.layer_3.weight .= w3
NN_Conservative_params.layer_3.bias   .= b3

scale_factor = semilatusRectum / (1 - eccentricity)
NN_components = (params = NN_Conservative_params, state = NN_Conservative_state, scale = scale_factor)

function NN_adapter(u)
    r = u[2]
    x = [r / NN_components.scale]

    out, _ = NN_Conservative(x, NN_components.params, NN_components.state)
    return (conservative = out, dissipative = (1.0,))
end 

function plot_metric_components(r_min, r_max)
    r_values = range(r_min, r_max, length = 40)

    f_tt_predicted = []
    f_rr_predicted = []
    
    for r in r_values
        output, _ = NN_Conservative([r / NN_components.scale], NN_components.params, NN_components.state)
        push!(f_tt_predicted, 1 - softplus(output[1]))
        push!(f_rr_predicted, 1 + softplus(output[2]))
    end

    g_tt_Schwarzschild = [-(1 .- 2 ./ r) for r in r_values]
    g_rr_Schwarzschild = [(1 .- 2 ./ r)^(-1) for r in r_values]

    g_tt_NN = [-1 * f_tt for f_tt in f_tt_predicted]
    g_rr_NN = [1 * f_rr for f_rr in f_rr_predicted]

    g_tt_Minkowski = [-1 for _ in f_tt_predicted]
    g_rr_Minkowski = [1 for _ in f_rr_predicted]

    g_tt_plots = plot(r_values, g_tt_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_tt_plots, r_values, g_tt_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")
    plot!(g_tt_plots, r_values, g_tt_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{tt}", title=L"g_{tt}(r)")

    g_rr_plots = plot(r_values, g_rr_Schwarzschild, lw=2, ls=:dash, color=:blue, label="Schwarzschild", legend=:topright)
    plot!(g_rr_plots, r_values, g_rr_NN, lw=2, color=:green, label="NN Prediction", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")
    plot!(g_rr_plots, r_values, g_rr_Minkowski, lw=2, color=:red, label="Minkowski", xlabel=L"r", ylabel=L"g_{rr}", title=L"g_{rr}(r)")
    
    print("L2 norm between g_rr_schwarzschild and g_rr_NN: ", norm(g_rr_Schwarzschild - g_rr_NN))
    print("\nL2 norm between g_rr_minkowski and g_rr_NN: ", norm(g_rr_Minkowski - g_rr_NN))

    return g_tt_plots, g_rr_plots
end

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    # Relativistc Forms
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )
  
    return [M, E, L]
end

function Schwarzschild(du, u, model_params, t)
    x = @view u[1:8]

    q = model_params[1]
    M = 1.0

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))

        p = [p_t, p_r, p_θ, p_ϕ]

        f = 1 - (2/r)

        # Base Metric: Schwarzschild Metric
        g = [
                -f 0 0 0;
                0 f^(-1) 0 0;
                0 0 1e-8 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = J * grad_H
    
    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = 0
    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end

function Schwarzschild_with_synthetic_grr(du, u, model_params, t)
    x = @view u[1:8]

    q = model_params[1]
    M = 1.0

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))

        p = [p_t, p_r, p_θ, p_ϕ]

        f = 1 - (2/r)

        # Base Metric: Schwarzschild Metric
        g = [
                -f 0 0 0;
                0 1.005*(r^2) 0 0;
                0 0 1e-8 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = J * grad_H
    
    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = 0
    du[1] = 1

    

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end

function Schwarzschild_with_NN(du, u, model_params, t;
                              NN=nothing, NN_params=nothing)
    x = @view u[1:8]

    q = model_params[1]

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))

        p = [p_t, p_r, p_θ, p_ϕ]

        f = 1 - (2/r)
        g_tt = -f
        g_rr = 1/f

        if NN !== nothing
            nn_out = NN(state_vec)
            raw_tt = nn_out.conservative[1]
            raw_rr = nn_out.conservative[2]

            f_tt = 1 - softplus(raw_tt)
            f_rr = 1 + softplus(raw_rr)

            g_tt = -(f_tt)
            g_rr = f_rr
        end

        # Base Metric: Schwarzschild Metric
        g = [
                g_tt 0 0 0;
                0 g_rr 0 0;
                0 0 r^2 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    Conservative = J * grad_H
    
    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = 0
    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end

function Schwarzschild_with_Dissipation(du, u, model_params, t)
    x = @view u[1:8]

    q = model_params[1]
    M = 1.0

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        f = (1 - (2/r))

        p = [p_t, p_r, p_θ, p_ϕ]

        f = 1 - (2/r)

        # Base Metric: Schwarzschild Metric
        g = [
                -f 0 0 0;
                0 f^(-1) 0 0;
                0 0 1e-8 0;
                0 0 0 r^2
            ]

        inv_g = Diagonal(1 ./ diag(g))
        H_schwarzschild = (1/2) * p' * inv_g * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    # Conservative Dynamics
    Conservative = J * grad_H

    # Dissipative Post-Newtonian Dynamics
    Dissipative = [0, 0, 0, 0, 0, 0, 0, -3.45e-1 * (x[2])^(-7/2)]
    
    du_dτ = Conservative + Dissipative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[9] = 0
    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8], du[9]]
end

function write_trajectory_txt(path::AbstractString, t::AbstractVector,
                              x::AbstractVector, y::AbstractVector)
    @assert length(t) == length(x) == length(y)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e %.18e\n", t[i], x[i], y[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function write_waveform_txt(path::AbstractString, t::AbstractVector, h::AbstractVector)
    @assert length(t) == length(h)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e\n", t[i], h[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function write_r_variable(path::AbstractString, t::AbstractVector, r::AbstractVector)
    @assert length(t) == length(r)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e\n", t[i], r[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function write_ϕ_variable(path::AbstractString, t::AbstractVector, ϕ::AbstractVector)
    @assert length(t) == length(ϕ)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e\n", t[i], ϕ[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function generate_training_data(semilatusRectum, eccentricity, mass_ratio, max_time, num_datapoints, dt_solver, type)
    
    @show type
    @assert type in ("schwarzschild", "synthetic_grr", "NN", "dissipation")

    tspan = (0, max_time)
    datasize = num_datapoints
    tsteps = range(tspan[1], tspan[2], length = datasize)
    dt_data = tsteps[2] - tsteps[1]
    print(dt_data, "\n")
    dt = dt_solver

    model_params = [mass_ratio] # Just the mass ratio, at least for now
    model_params_2021 = [semilatusRectum, 1.0, eccentricity]
    mass1 = 1.0/(1.0+mass_ratio)
    mass2 = mass_ratio/(1.0+mass_ratio)

    E0, L0 = pe_2_EL(semilatusRectum, eccentricity)[2:3] # Schwarzschild ICs
    R = semilatusRectum / (1 - eccentricity)
    
    u0 = [0.0, R, pi/2, 0.0, -E0, 0.0, 0.0, L0, 0.0]
    u0_2021 = [pi, 0.0]

    ODE_model = if type == "schwarzschild"
        (du, u, p, t) -> Schwarzschild(du, u, model_params, t)
    elseif type == "synthetic_grr"
        (du, u, p, t) -> Schwarzschild_with_synthetic_grr(du, u, model_params, t)
    elseif type == "NN"
        (du, u, p, t) -> Schwarzschild_with_NN(du, u, model_params, t,
                                               NN = NN_adapter,
                                               NN_params = NN_components)
    elseif type == "dissipation"
        (du, u, p, t) -> Schwarzschild_with_Dissipation(du, u, model_params, t)
    else
        error("Come on, man")
    end

    function ODE_model_2021(u, p, t)
        du = RelativisticOrbitModel(u, model_params_2021, t)
        return du
    end

    # CREATE & SOLVE ODE PROBLEM (2025)
    prob = ODEProblem(ODE_model, u0, tspan)
    soln = Array(solve(prob, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
    orbit = soln2orbit(soln)
    blackHole_r1, blackHole_r2 = one2two(orbit, mass1, mass2)
    h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)

    # CREATE & SOLVE ODE MODEL (2021)
    prob_2021 = ODEProblem(ODE_model_2021, u0_2021, tspan, model_params_2021)
    soln_2021 = Array(solve(prob_2021, Tsit5(), saveat = tsteps, dt = dt, adaptive = false))
    orbit_2021 = soln2orbit_2021(soln_2021, model_params_2021)
    blackHole_r1_2021, blackHole_r2_2021 = one2two(orbit_2021, mass1, mass2)
    h_plus_true_2021, _ = h_22_strain_two_body(dt_data, blackHole_r1_2021, mass1, blackHole_r2_2021, mass2)

    # ONE-BODY ORBIT PLOT
    oneBodyOrbit = plot(orbit[1,:], orbit[2,:], aspect_ratio=:equal, linewidth = 2, label = "One-Body Orbit (2025)")
    plot!(oneBodyOrbit, orbit_2021[1,:], orbit_2021[2,:], aspect_ratio=:equal, linewidth = 2, label = "One-Body Orbit (2021)")
    display(oneBodyOrbit)

    # # TWO-BODY ORBIT PLOT
    orbits = plot(blackHole_r1[1,:], blackHole_r1[2,:], aspect_ratio=:equal, linewidth = 2, label = "Black Hole 1")
    plot!(orbits, blackHole_r2[1,:], blackHole_r2[2,:], aspect_ratio=:equal, linewidth = 2, legend=:topleft, label = "Black Hole 2")
    display(orbits)

    # PREDICTED METRIC PLOT
    g_tt_plot, g_rr_plot = plot_metric_components(7, 16)
    display(g_tt_plot)
    display(g_rr_plot)

    # WAVEFORM PLOT
    h₊_plot = plot(tsteps, h_plus_true, linewidth = 2, legend=:topleft, label = L"h_+ (2025)")
    plot!(h₊_plot, tsteps, h_plus_true_2021, linewidth = 2, legend=:topleft, label = L"h_+ (2021)")
    display(h₊_plot)

    print("\nL2 Norm between 2025 and 2021 h_plus: ", norm(h_plus_true - h_plus_true_2021, 2), "\n")

    x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
    x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]
    hplus  = h_plus_true
    hcross = h_cross_true

    t_vec = collect(tsteps)
    t0 = t_vec[1]
    t_out = t_vec .- t0  

    write_trajectory_txt("./input/trajectoryA_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, x1, y1)
    write_trajectory_txt("./input/trajectoryB_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, x2, y2)
    write_waveform_txt("./input/waveform_real_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, hplus)
    write_waveform_txt("./input/waveform_imag_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, hcross)
    write_r_variable("./input/r_variable_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, soln[2, :])
    write_ϕ_variable("./input/phi_variable_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, soln[4, :])
end