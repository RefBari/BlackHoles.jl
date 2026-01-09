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

include("2021EquationFunctions.jl")
include("Orbit2Waveform.jl")
include("TrainingDataFunctions.jl")

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    # Relativistc Forms
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )
  
    return [M, E, L]
  end


function Schwarzschild(du, u, model_params, t;
                              NN=nothing, NN_params=nothing)
    x = @view u[1:8]

    q = model_params[1]
    M = 1.0

    if isnothing(NN)
        conservative_corrections = [0]
        dissipative_corrections = 1.0
    else
        nn_outputs = NN(u, NN_params)
        conservative_corrections = nn_outputs.conservative
        dissipative_corrections = nn_outputs.dissipative[1]
    end

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

        H_schwarzschild = (1/2) * p' * inv(g) * p

        return H_schwarzschild # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    # Conservative Dynamics
    Conservative = J * grad_H
    
    # Dissipative Dynamics
    Dissipative = [0, 0, 0, 0, 0, 0, 0, -0.0103/x[2]]

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

function generateDissipativeTraining(semilatusRectum, eccentricity, mass_ratio, max_time, num_datapoints, dt_solver)
    logocolors = Colors.JULIA_LOGO_COLORS
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

    function ODE_model(du, u, p, t)
        du = Schwarzschild(du, u, model_params, t)
        return du
    end

    function ODE_model_2021(u, p, t)
        du = RelativisticOrbitModel(u, model_params_2021, t)
        return du
    end

    prob = ODEProblem(ODE_model, u0, tspan)
    soln = Array(solve(prob, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
    orbit = soln2orbit(soln)
    blackHole_r1, blackHole_r2 = one2two(orbit, mass1, mass2)
    h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)

    oneBodyOrbit = plot(orbit[1,:], orbit[2,:], aspect_ratio=:equal, linewidth = 2, label = "One-Body Orbit", title = latexstring("\$\\ p = {$(semilatusRectum)}, e = {$(eccentricity)}\$"))

    orbits = plot(blackHole_r1[1,:], blackHole_r1[2,:], aspect_ratio=:equal, linewidth = 2, label = "Black Hole 1", title = latexstring("\$\\ p = {$(semilatusRectum)}, e = {$(eccentricity)}\$"))
    plot!(orbits, blackHole_r2[1,:], blackHole_r2[2,:], aspect_ratio=:equal, linewidth = 2, legend=:topleft, label = "Black Hole 2")

    h₊_plot = plot(tsteps, h_plus_true, color = logocolors.red, linewidth = 2, legend=:topleft, label = L"h_+", title = "Gravitational Wave")
    hx_plot = plot(tsteps, h_cross_true, linewidth = 2, legend=:topleft, label = L"h_×")

    z = collect(tsteps)

    orbit_3d = plot3d(
        blackHole_r1[1,:], blackHole_r1[2,:], z,
        linewidth = 2,
        label = "Black Hole 1 Trajectory",
        legend = :topleft,
        title = "3D Binary Black Hole Orbits"
    )
    plot3d!(
        orbit_3d,
        blackHole_r2[1,:], blackHole_r2[2,:], z,
        linewidth = 2,
        label = "Black Hole 2 Trajectory"
    )

    iron_man_dashboard = plot(oneBodyOrbit, orbits, h₊_plot, orbit_3d, layout = (2, 2), 
                                    size = (1000, 1000), 
                                    plot_title = "Dissipative Training Data",
                                    plot_titlevspan = 0.05, 
                                    top_margin = 10mm,
                                    bottom_margin = 10mm,
                                    left_margin = 10mm,
                                    right_margin = 10mm)
    display(iron_man_dashboard)

    x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
    x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]
    hplus  = h_plus_true
    hcross = h_cross_true

    t_vec = collect(tsteps)
    t0 = t_vec[1]
    t_out = t_vec .- t0  

    write_trajectory_txt("./input/dissipative_trajectoryA_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, x1, y1)
    write_trajectory_txt("./input/dissipative_trajectoryB_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, x2, y2)
    write_waveform_txt("./input/dissipative_waveform_real_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, hplus)
    write_waveform_txt("./input/dissipative_waveform_imag_Schwarzschild_p"*string(semilatusRectum)*"_e"*string(eccentricity)*".txt", t_out, hcross)
end