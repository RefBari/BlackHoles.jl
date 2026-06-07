
# ======================================================================
# compactTrainingFunctions.jl
# Create Kerr Geodesics from r_min, r_max, and θ_min
# ======================================================================

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
using Noise
using GeometryBasics
using Interpolations
using ColorTypes: RGBAf, RGBAf0
using GeometryBasics: Point3f
using NLsolve
using ForwardDiff
using DifferentialEquations
using Plots
# using WGLMakie
using CairoMakie   # swap to CairoMakie here to go back to static
using FFTW
include("2021EquationFunctions.jl")
include("Orbit2Waveform.jl")

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

function Kerr(du, u, model_params, t; M = 1.0, a = 0.9)
    x = @view u[1:8]
    q = model_params[1]

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        p = [p_t, p_r, p_θ, p_ϕ]

        Σ = r^2 + a^2 * cos(θ)^2
        Δ = r^2 - 2*M*r + a^2

        g_tt = -(1 - (2*M*r)/ Σ )
        g_rr = Σ / Δ
        g_θθ = Σ
        g_ϕϕ = (r^2 + a^2 + (2*M*r*a^2*sin(θ)^2) / Σ) * sin(θ)^2
        g_tϕ = - (2*M*r*a*sin(θ)^2) / Σ
        g_ϕt = g_tϕ

        # Base Metric: Kerr Metric
        g = [
                g_tt 0 0 g_tϕ;
                0 g_rr 0 0;
                0 0 g_θθ 0;
                g_ϕt 0 0 g_ϕϕ
            ]

        inv_g = inv(g)
        H_kerr = (1/2) * p' * inv_g * p

        return H_kerr # Returns equations of motion in PROPER time
    end

    grad_H = ForwardDiff.gradient(H, x)

    J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

    du_dτ = J * grad_H

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8]]
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

# Create function to write 8-element solution state vector to file 
function write_solution_txt(path::AbstractString, t::AbstractVector, u::AbstractMatrix)
    @assert length(t) == size(u, 2)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e ", t[i])
            for j in 1:size(u, 1)
                @printf(io, "%.18e ", u[j, i])
            end
            @printf(io, "\n")
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function solve_ELQ(rₚ, rₐ, θmin; M = 1.0, a = 0.9, μ = 1.0)
    Δₚ = rₚ^2 - 2*M*rₚ + a^2
    Δₐ = rₐ^2 - 2*M*rₐ + a^2

    function equations!(F, x)
        E, Lz = x[1], x[2]
        Q_from_rₚ = (((rₚ^2 + a^2) * E - a * Lz)^2 - Δₚ * ((Lz-a*E)^2 + μ^2 * rₚ^2) ) / Δₚ

        F[1] = ((rₐ^2 + a^2) * E - a * Lz)^2 - Δₐ * ( (Lz - a*E)^2 + μ^2 * rₐ^2 + Q_from_rₚ )
        F[2] = Q_from_rₚ - cos(θmin)^2 * (a^2 * (μ^2 - E^2) + Lz^2 / sin(θmin)^2 )
    end

    # Initial Guesses
    E0 = sqrt( 1 - 2 * M / (rₚ + rₐ) )
    L0 = sqrt( M * (rₚ + rₐ) / 2 )
    sol = nlsolve(equations!, [E0, L0]; autodiff=:forward, ftol = 1e-12)

    E, Lz = sol.zero
    Δₚ_val = rₚ^2 - 2*M*rₚ + a^2
    Q = ( ((rₚ^2 + a^2) * E - a * Lz)^2 - Δₚ_val *  ( (Lz - a*E)^2 + μ^2 * rₚ^2 ) ) / Δₚ_val

    return E, Lz, Q
end

function u0_from_geometry(rp, ra, θmin; M = 1.0, a = 0.9, μ = 1.0)
    E, Lz, Q = solve_ELQ(rp, ra, θmin; M = M, a = a, μ = μ)
    return [0.0, rp, θmin, 0.0, -E, 0.0, 0.0, Lz], E, Lz, Q
end

function solve_kerr(u0; M=1.0, a=0.9, λmax=1e3, saveat=0.1,
                           reltol=1e-10, abstol=1e-10)

    kerr_ode! = (du, u, p, t) -> Kerr(du, u, [1.0], t; M=p[1], a=p[2])
    prob = ODEProblem(kerr_ode!, u0, (0.0, λmax), (M, a))
    sol  = solve(prob, Vern9(); reltol=reltol, abstol=abstol, saveat=saveat)

    return sol
end

function add_turning_point_circles!(ax; r_vals, θ_vals, a=0.9, zplane=0.0,
                                    color=:black, npts=300)

    ϕc = range(0, 2π, length=npts)

    # Actual projected xy radius of the plotted geodesic
    ρxy = @. sqrt(r_vals^2 + a^2) * sin(θ_vals)

    # Use the displayed projection envelope
    ρp = minimum(ρxy)
    ρa = maximum(ρxy)

    xp = ρp .* cos.(ϕc)
    yp = ρp .* sin.(ϕc)
    zp = fill(zplane, length(ϕc))

    xa = ρa .* cos.(ϕc)
    ya = ρa .* sin.(ϕc)
    za = fill(zplane, length(ϕc))

    # Apoapsis-like outer projected envelope: solid
    CairoMakie.lines!(ax, xa, ya, za;
        color=color, linewidth=2.0)

    # Periapsis-like inner projected envelope: dashed
    CairoMakie.lines!(ax, xp, yp, zp;
        color=color, linewidth=1.5, linestyle=:dash)

    return nothing
end

function plot_geodesic_publication_fast(soln; M = 1.0, a = 0.9,
                                        projection_z = -50.0,
                                        figsize = (750, 950),
                                        stride = 1,
                                        θmin)

    r = soln[2, 1:stride:end]
    θ = soln[3, 1:stride:end]
    ϕ = soln[4, 1:stride:end]

    x = @. r * sin(θ) * cos(ϕ)
    y = @. r * sin(θ) * sin(ϕ)
    z = @. r * cos(θ)

    rp = minimum(r)
    ra = maximum(r)

    ρmax = sqrt(ra^2 + a^2)
    Rxy = 1.30 * ρmax   # a little padding

    zlo = min(minimum(z), projection_z) - 4
    zhi = maximum(z) + 4

    fig = CairoMakie.Figure(size = figsize, backgroundcolor = :white)

    ax = CairoMakie.Axis3(fig[1, 1];
        aspect = :data,
        xlabel = L"x",
        ylabel = L"y",
        zlabel = L"z",
        azimuth = 1.3π,
        elevation = 0.25π,
        protrusions = 20,
        viewmode = :fit,
        limits = ((-Rxy, Rxy), (-Rxy, Rxy), (zlo, zhi)),
    )

    ax.xgridcolor = RGBAf(0, 0, 0, 0.03)
    ax.ygridcolor = RGBAf(0, 0, 0, 0.03)
    ax.zgridcolor = RGBAf(0, 0, 0, 0.03)
    
    # pad_xy = 0.12 * ρmax
    # xlims!(ax, -(ρmax + pad_xy),  ρmax + pad_xy)
    # ylims!(ax, -(ρmax + pad_xy),  ρmax + pad_xy)
    
    add_turning_point_circles!(ax;
        r_vals = r,
        θ_vals = θ,
        a = a,
        zplane = projection_z
    )
        
    rlims = extrema(r)

    CairoMakie.lines!(ax, x, y, z;
        color = r,
        colormap = :cool,
        colorrange = rlims,
        linewidth = 1.5
    )

    CairoMakie.lines!(ax, x, y, fill(projection_z, length(x));
        color = r,
        colormap = :cool,
        colorrange = rlims,
        linewidth = 0.8,
        alpha = 0.3
    )

    rH = M + sqrt(M^2 - a^2)
    θg = range(0, π,  length = 20)
    ϕg = range(0, 2π, length = 40)
    Xh = [sqrt(rH^2 + a^2) * sin(th) * cos(ph) for th in θg, ph in ϕg]
    Yh = [sqrt(rH^2 + a^2) * sin(th) * sin(ph) for th in θg, ph in ϕg]
    Zh = [rH * cos(th) for th in θg, ph in ϕg]

    CairoMakie.surface!(ax, Xh, Yh, Zh;
        color = :black,
        alpha = 0.9,
        shading = NoShading
    )

    CairoMakie.Colorbar(fig[2, 1],
        colormap = :cool,
        limits = rlims,
        label = L"r / r_g",
        vertical = false,
        width = Relative(0.6),
        tellwidth = false
    )

    return fig
end

function generate_training_data(rp, ra, θmin, BH_spin, mass_ratio, max_time, num_datapoints, dt_solver, type, noise = 0.0)
    
    @show type
    @assert type in ("kerr")

    tspan = (0, max_time)
    datasize = num_datapoints
    tsteps = range(tspan[1], tspan[2], length = datasize)
    dt_data = tsteps[2] - tsteps[1]
    print(dt_data, "\n")
    dt = dt_solver
    
    model_params = [mass_ratio] # Just the mass ratio, at least for now
    mass1 = mass_ratio/(1.0+mass_ratio)
    mass2 = 1.0/(1.0+mass_ratio)
    
    u0_kerr, E, Lz, Q = u0_from_geometry(rp, ra, θmin; M=1.0, a=BH_spin, μ=1.0)

    ODE_model = if type == "kerr"
        (du, u, p, t) -> Kerr(du, u, model_params, t; M = 1.0, a = BH_spin)
    else
        error("Come on, man")
    end

    # CREATE & SOLVE ODE PROBLEM (2025)
    prob = ODEProblem(ODE_model, u0_kerr, tspan)
    soln = Array(solve(prob, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
    orbit = soln2orbit(soln, a = BH_spin)
    blackHole_r1, blackHole_r2 = one2two(orbit, mass1, mass2)
    h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)
    
    println("dt_data = $dt_data")
    println("orbit x range: $(extrema(orbit[1,:]))")
    println("any NaN in orbit? $(any(isnan, orbit))")
    println("any NaN in h+? $(any(isnan, h_plus_true))")
    println("h+ range: $(extrema(h_plus_true))")
    println("length(tsteps)    = $(length(tsteps))")
    println("length(h_plus_true) = $(length(h_plus_true))")
    println("tsteps range: $(first(tsteps)) → $(last(tsteps))")

    # WAVEFORM PLOT
    fig_wave = CairoMakie.Figure(size=(900, 400))
    ax_wave = CairoMakie.Axis(fig_wave[1, 1];
                      xlabel = L"Time ($M$)",
                      ylabel = L"$h(t)$",
                      title = "Gravitational Wave")
    CairoMakie.lines!(ax_wave, collect(tsteps), h_plus_true;
              linewidth = 2, color=:blue)
    display(fig_wave)

    fig = plot_geodesic_publication_fast(soln; M = 1.0, a = BH_spin, θmin)
    display(fig)

    x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
    x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]
    hplus  = h_plus_true
    hcross = h_cross_true

    t_vec = collect(tsteps)
    t0 = t_vec[1]
    t_out = t_vec .- t0  

    write_solution_txt("./input/solution_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, soln)
    write_trajectory_txt("./input/trajectoryA_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, x1, y1)
    write_trajectory_txt("./input/trajectoryB_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, x2, y2)
    write_waveform_txt("./input/waveform_real_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, hplus)
    write_waveform_txt("./input/waveform_imag_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, hcross)
end
