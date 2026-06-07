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

function Kerr(du, u, model_params, t; M = 1.0, a = 0.9)
    x = @view u[1:8]
    q = model_params[1]

    function H(state_vec)
        t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

        p = [p_t, p_r, p_θ, p_ϕ]

        ρ = r^2 + a^2 * cos(θ)^2
        Δ = r^2 - 2*M*r + a^2

        g_tt = -(1 - (2*M*r)/ ρ )
        g_rr = ρ / Δ
        g_θθ = ρ
        g_ϕϕ = (r^2 + a^2 + (2*M*r*a^2*sin(θ)^2) / ρ) * sin(θ)^2
        g_tϕ = - (2*M*r*a*sin(θ)^2) / ρ
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

    Conservative = J * grad_H
    
    du_dτ = Conservative

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)

    @inbounds for i = 1:8
        du[i] = du_dτ[i] * dτ_dt
    end

    du[1] = 1

    return [du[1], du[2], du[3], du[4], du[5], du[6], du[7], du[8]]
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
    Dissipative = [0, 0, 0, 0, 0, 0, 0, -3.3e-1 * (x[2])^(-7/2)]
    #           = [t, r, θ, ϕ, pₜ, pᵣ, p_θ, p_ϕ]
    
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

function makie_geodesic_3d(sol; M=1.0, a=0.9, show_horizon=true, size=(900,700), savepath=nothing)
    r = sol[2, :];  θ = sol[3, :];  ϕ = sol[4, :]

    x = @. r * sin(θ) * cos(ϕ)
    y = @. r * sin(θ) * sin(ϕ)
    z = @. r * cos(θ)

    fig = CairoMakie.Figure(size=size)
    ax  = CairoMakie.Axis3(fig[1,1], aspect=:data, xlabel="x", ylabel="y", zlabel="z")

    CairoMakie.lines!(ax, x, y, z; linewidth=2)
    CairoMakie.scatter!(ax, [x[1]],   [y[1]],   [z[1]];   markersize=10)
    CairoMakie.scatter!(ax, [x[end]], [y[end]], [z[end]];  markersize=14, marker=:star5)

    if show_horizon
        rH = M + sqrt(M^2 - a^2)
        θg = range(0, π;  length=25)
        ϕg = range(0, 2π; length=50)
        X = [rH*sin(th)*cos(ph) for th in θg, ph in ϕg]
        Y = [rH*sin(th)*sin(ph) for th in θg, ph in ϕg]
        Z = [rH*cos(th)         for th in θg, ph in ϕg]
        CairoMakie.surface!(ax, X, Y, Z; alpha=0.15)   # transparency= is not a valid kwarg in Makie — use alpha= only
    end

    if savepath !== nothing
        CairoMakie.save(savepath, fig)
    end
    return fig
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
                                        projection_z = -20.0,
                                        figsize = (750, 950),
                                        stride = 1,
                                        θmin)

    r = soln[2, 1:stride:end]
    θ = soln[3, 1:stride:end]
    ϕ = soln[4, 1:stride:end]

    x = @. sqrt(r^2 + a^2) * sin(θ) * cos(ϕ)
    y = @. sqrt(r^2 + a^2) * sin(θ) * sin(ϕ)
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
    
    # add_turning_point_circles!(ax;
    #     r_vals = r,
    #     θ_vals = θ,
    #     a = a,
    #     zplane = projection_z
    # )
        
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

function plot_waveform_fft(t, h)
    N = length(h)
    dt = t[2] - t[1]
    
    H = fft(h)
    freqs = (0:N-1) ./ (N * dt)
    amp = abs.(H)

    half = 1:div(N, 2)

    fig = CairoMakie.Figure(size=(900, 400))
    ax = CairoMakie.Axis(fig[1, 1];
                    xlabel = "Frequency (Hz)",
                    ylabel = "Amplitude",
                    title = "Gravitational Wave FFT")
    CairoMakie.lines!(ax, freqs[half], amp[half];
              linewidth = 2, color=:blue)
    display(fig)

    return freqs[half], amp[half], fig
end

function generate_training_data(rp, ra, θmin, BH_spin, mass_ratio, max_time, num_datapoints, dt_solver, type, noise = 0.0)
    
    @show type
    @assert type in ("schwarzschild", "kerr", "synthetic_grr", "NN", "dissipation")

    tspan = (0, max_time)
    datasize = num_datapoints
    tsteps = range(tspan[1], tspan[2], length = datasize)
    dt_data = tsteps[2] - tsteps[1]
    print(dt_data, "\n")
    dt = dt_solver
    
    model_params = [mass_ratio] # Just the mass ratio, at least for now
    # model_params_2021 = [semilatusRectum, 1.0, eccentricity]
    mass1 = mass_ratio/(1.0+mass_ratio)
    mass2 = 1.0/(1.0+mass_ratio)

    # E0, L0 = pe_2_EL(semilatusRectum, eccentricity)[2:3] # Schwarzschild ICs
    # R = semilatusRectum / (1 - eccentricity)
    
    # u0 = [0.0, R, pi/2, 0.0, -E0, 0.0, 0.0, L0, 0.0]
    # u0_2021 = [pi, 0.0]
    u0_kerr, E, Lz, Q = u0_from_geometry(rp, ra, θmin; M=1.0, a=BH_spin, μ=1.0)

    # u0_ZAMO = copy(u0_kerr)
    # u0_ZAMO[8] = 0.0 # Set p_ϕ = 0

    ODE_model = if type == "schwarzschild"
        (du, u, p, t) -> Schwarzschild(du, u, model_params, t)
    elseif type == "kerr"
        (du, u, p, t) -> Kerr(du, u, model_params, t; M = 1.0, a = BH_spin)
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
    prob = ODEProblem(ODE_model, u0_kerr, tspan)
    soln = Array(solve(prob, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
    orbit = soln2orbit(soln, a = BH_spin)
    blackHole_r1, blackHole_r2 = one2two(orbit, mass1, mass2)
    h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)
    
    t_vals = soln[1, :]
    r_vals = soln[2, :]
    θ_vals = soln[3, :]
    ϕ_vals = soln[4, :]

    Ω_numerical = diff(ϕ_vals) ./ diff(t_vals)

    M = 1.0
    a = BH_spin

    Ω_theory = map(zip(r_vals[1:end-1], θ_vals[1:end-1])) do (r, θ)
        Σ = r^2 + a^2 * cos(θ)^2
        g_tϕ = - (2*M*r/Σ) * a * sin(θ)^2
        g_ϕϕ = (r^2 + a^2 + (2*M*r*a^2*sin(θ)^2) / Σ) * sin(θ)^2
        - g_tϕ / g_ϕϕ
    end

    println("dt_data = $dt_data")
    println("orbit x range: $(extrema(orbit[1,:]))")
    println("any NaN in orbit? $(any(isnan, orbit))")
    println("any NaN in h+? $(any(isnan, h_plus_true))")
    println("h+ range: $(extrema(h_plus_true))")
    println("length(tsteps)    = $(length(tsteps))")
    println("length(h_plus_true) = $(length(h_plus_true))")
    println("tsteps range: $(first(tsteps)) → $(last(tsteps))")

    # # CREATE & SOLVE ODE MODEL (2021)
    # prob_2021 = ODEProblem(ODE_model_2021, u0_2021, tspan, model_params_2021)
    # soln_2021 = Array(solve(prob_2021, Tsit5(), saveat = tsteps, dt = dt, adaptive = false))
    # orbit_2021 = soln2orbit_2021(soln_2021, model_params_2021)
    # blackHole_r1_2021, blackHole_r2_2021 = one2two(orbit_2021, mass1, mass2)
    # h_plus_true_2021, _ = h_22_strain_two_body(dt_data, blackHole_r1_2021, mass1, blackHole_r2_2021, mass2)

    # ONE-BODY ORBIT PLOT
    # stopOrbit = 200
    # oneBodyOrbit = plot(orbit[1,1:stopOrbit], orbit[2,1:stopOrbit], aspect_ratio=:equal, linewidth = 2, label = "One-Body Orbit (2025)")
    # # plot!(oneBodyOrbit, orbit_2021[1,1:stopOrbit], orbit_2021[2,1:stopOrbit], aspect_ratio=:equal, linewidth = 2, label = "One-Body Orbit (2021)")
    # display(oneBodyOrbit)

    # # # TWO-BODY ORBIT PLOT
    # orbits = plot(blackHole_r1[1,:], blackHole_r1[2,:], aspect_ratio=:equal, linewidth = 2, label = "Black Hole 1")
    # plot!(orbits, blackHole_r2[1,:], blackHole_r2[2,:], aspect_ratio=:equal, linewidth = 2, legend=:topleft, label = "Black Hole 2")
    # display(orbits)

    # # PREDICTED METRIC PLOT
    # g_tt_plot, g_rr_plot = plot_metric_components(7, 16)
    # display(g_tt_plot)
    # display(g_rr_plot)

    # WAVEFORM PLOT
    fig_wave = CairoMakie.Figure(size=(900, 400))
    ax_wave = CairoMakie.Axis(fig_wave[1, 1];
                      xlabel = L"Time ($M$)",
                      ylabel = L"$h(t)$",
                      title = "Gravitational Wave")
    CairoMakie.lines!(ax_wave, collect(tsteps), h_plus_true;
              linewidth = 2, color=:blue)
    # CairoMakie.axislegend(ax_wave)
    display(fig_wave)

    # Oscillation in Radial (r), Polar (θ), and Azimuthal (ϕ) Coordinates
    fig_radial_oscillations = CairoMakie.Figure(size=(900, 400))
    ax_wave = CairoMakie.Axis(fig_radial_oscillations[1, 1];
                      xlabel = "Time (M)",
                      ylabel = "Value",
                      title = "Radial (r)")
    CairoMakie.lines!(ax_wave, collect(tsteps), r_vals;
              linewidth = 2, color=:red, label = L"r(t)")
    CairoMakie.axislegend(ax_wave)
    display(fig_radial_oscillations)

    # ZAMO Plot
    # ZAMO_Plot = CairoMakie.Figure(size=(900, 400)) 
    # ax_ZAMO = CairoMakie.Axis(ZAMO_Plot[1, 1];
    #                   xlabel = L"Time ($M$)",
    #                   ylabel = L"Ω",
    #                   title = "ZAMO Orbital Frequency")
    # CairoMakie.lines!(ax_ZAMO, collect(tsteps[1:end-1]), Ω_numerical;
    #           linewidth = 2, color=:blue, label = L"Numerical $Ω$")
    # CairoMakie.lines!(ax_ZAMO, collect(tsteps[1:end-1]), Ω_theory;
    #           linewidth = 2, color=:orange, linestyle=:dash, label = L"Theoretical $Ω$")
    # CairoMakie.axislegend(ax_ZAMO)
    # display(ZAMO_Plot)

    # Polar Oscillations Plot
    fig_polar_oscillations = CairoMakie.Figure(size=(900, 400))
    ax_wave = CairoMakie.Axis(fig_polar_oscillations[1, 1];
                      xlabel = "Time (M)",
                      ylabel = "Value",
                      title = "Polar (θ)")
    CairoMakie.lines!(ax_wave, collect(tsteps), θ_vals;
              linewidth = 2, color=:red, label = L"θ(t)")
    CairoMakie.axislegend(ax_wave)
    display(fig_polar_oscillations)

    fig_azimuthal_oscillations = CairoMakie.Figure(size=(900, 400))
    ax_wave = CairoMakie.Axis(fig_azimuthal_oscillations[1, 1];
                      xlabel = "Time (M)",
                      ylabel = "Value",
                      title = "Azimuthal (ϕ)")
    CairoMakie.lines!(ax_wave, collect(tsteps), ϕ_vals;
              linewidth = 2, color=:red, label = L"ϕ(t)")
    CairoMakie.axislegend(ax_wave)
    display(fig_azimuthal_oscillations)

    # h₊_plot = Plots.plot(tsteps, h_plus_true, linewidth = 2, legend=:topleft, label = L"h_+ (2025)")
    # plot!(h₊_plot, tsteps, h_plus_true_2021, linewidth = 2, legend=:topleft, label = L"h_+ (2021)")

    # sol = solve_kerr(u0_kerr; M = 1.0, a = BH_spin, λmax = max_time, saveat = 0.2)
    # fig = makie_geodesic_3d(sol; M=1.0, a = BH_spin)
    # display(fig)
    fig = plot_geodesic_publication_fast(soln; M = 1.0, a = BH_spin, θmin)
    display(fig)

    freqs_plus, amp_plus, fig_fft_plus = plot_waveform_fft(tsteps, h_plus_true)
    display(fig_fft_plus)

    # print("\nL2 Norm between 2025 and 2021 h_plus: ", norm(h_plus_true - h_plus_true_2021, 2), "\n")

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
    # write_r_variable("./input/r_variable_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, soln[2, :])
    # write_ϕ_variable("./input/phi_variable_Kerr_rp"*string(rp)*"_ra"*string(ra)*"noise"*string(noise)*"_massRatio"*string(mass_ratio)*"spin_"*string(BH_spin)*"inclination_"*string(θmin)*"_time"*string(max_time)*".txt", t_out, soln[4, :])
end

