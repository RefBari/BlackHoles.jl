include("2021CodeBase.jl")
include("GaussianModel.jl")

# SciML Tools
import OrdinaryDiffEq as ODE
import ModelingToolkit as MTK
import DataDrivenDiffEq
import SciMLSensitivity as SMS
import DataDrivenSparse
import Optimization as OPT
import OptimizationOptimisers
import OptimizationOptimJL
using Printf

# Standard Libraries
import LinearAlgebra
import Statistics

# External Libraries
import ComponentArrays
import Lux
import Zygote
import Plots
import StableRNGs
import DataFrames
import CSV
import LineSearches
using Plots
using ForwardDiff
using Lux
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
using Optim
using Optimization
using OptimizationOptimisers
using LaTeXStrings

Plots.gr()

# Set a random seed for reproducible behaviour
rng = StableRNGs.StableRNG(1111)

mass_ratio = 1       # test particle (has mass_ratio = 0)

u0 = Float64[pi, 0.0]
datasize = 300
tspan = (0.0f0, 5.0f3)   # timespace for GW waveform
tsteps = range(tspan[1], tspan[2], length = datasize)  # time at each timestep
dt_data = tsteps[2] - tsteps[1]
dt = 10
model_params = [20.0, 1.0, 0.5]; # p, M, e

mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

r0 = model_params[1]/(1 - model_params[3])
E0, L0 = pe_2_EL(model_params[1], model_params[3])[2:3]
u0_Schwarzschild = Float64[
                            0.0,    # t
                            r0,     # r
                            pi/2,   # θ
                            0.0,    # ϕ
                            -E0,    # p_t
                            0.0,    # p_r
                            0.0,    # p_θ
                            L0,     # p_ϕ
                            0.0
                        ]    # initial conditions

EOB_prob = ODE.ODEProblem(RelativisticOrbitModel, u0, tspan, model_params)
EOB_soln = Array(ODE.solve(EOB_prob, ODE.RK4(), saveat = tsteps, dt = dt, adaptive = false))
EOB_orbit = soln2orbit(EOB_soln, model_params)
EOB_waveform = compute_waveform(dt_data, EOB_soln, mass_ratio, model_params)[1]

Schwarzschild_prob = ODE.ODEProblem(GENERIC_Schwarzschild, u0_Schwarzschild, tspan, model_params)
Schwarschild_soln = Array(ODE.solve(Schwarzschild_prob, ODE.RK4(), saveat = tsteps, dt = dt, adaptive = false))
Schwarzschild_orbit = soln2orbit_Schwarzschild(Schwarschild_soln, model_params)
Schwarzschild_waveform = compute_waveform_schwarzschild(dt_data, Schwarschild_soln, mass_ratio, model_params)[1]

Newtonian_prob = ODE.ODEProblem(NewtonianOrbitModel, u0, tspan, model_params)
Newtonian_soln = Array(ODE.solve(Newtonian_prob, ODE.RK4(), saveat = tsteps, dt = dt, adaptive = false))
Newtonian_orbit = soln2orbit(Newtonian_soln, model_params)
Newtonian_waveform = compute_waveform(dt_data, Newtonian_soln, mass_ratio, model_params)[1]

plotTo = 300
plot(EOB_orbit[1, 1:plotTo], EOB_orbit[2, 1:plotTo], label = "True Orbit")
plot!(Newtonian_orbit[1, 1:plotTo], Newtonian_orbit[2, 1:plotTo], label = "Newtonian Orbit")
plot!(Schwarzschild_orbit[1, 1:plotTo], Schwarzschild_orbit[2, 1:plotTo], label = "Schwarzschild Orbit", xlabel = L"x", ylabel = L"y")

plot(tsteps[1:plotTo], real(EOB_waveform[1:plotTo]), label = "True Waveform")
plot!(tsteps[1:plotTo], real(Newtonian_waveform[1:plotTo]), label = "Newtonian Waveform")
plot!(tsteps[1:plotTo], real(Schwarzschild_waveform[1:plotTo]), label = "Schwarzschild Waveform", xlabel = "Time", ylabel = L"h_+")

blackHole_r1, blackHole_r2 = one2two(Schwarzschild_orbit, mass1, mass2)
h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)

plot!(tsteps[1:plotTo], h_plus_true[1:plotTo], label = "h₊_true")

x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]


plot(x1[1:plotTo], y1[1:plotTo], aspect_ratio=:equal)
plot!(x2[1:plotTo], y2[1:plotTo], aspect_ratio=:equal)

hplus  = h_plus_true
hcross = h_cross_true

t_out = collect(tsteps)

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

write_trajectory_txt("./input/trajectoryA_Schwarzschild_p20_e0p5.txt", t_out, x1, y1)
write_trajectory_txt("./input/trajectoryB_Schwarzschild_p20_e0p5.txt", t_out, x2, y2)
write_waveform_txt("./input/waveform_real_Schwarzschild_p20_e0p5.txt", t_out, hplus)
write_waveform_txt("./input/waveform_imag_Schwarzschild_p20_e0p5.txt", t_out, hcross)

mass_ratios = [0.1, 0.25, 0.5, 1.0]  # q = m1/m2 ≤ 1

waveforms = Dict{Float64, Vector{Float64}}()

for q in mass_ratios
    hplus_EOB, _ = compute_waveform(dt_data, EOB_soln, q, model_params)
    waveforms[q] = hplus_EOB
end

# Plot all waveforms normalized by their own max to see shape equality
massRatioPlots = plot(tsteps, waveforms[mass_ratios[1]], label = "q=$(mass_ratios[1])", xlabel = L"t", ylabel = L"h_+")

for q in mass_ratios[2:end]
    plot!(tsteps, waveforms[q],
          label = "q=$q")
end

display(massRatioPlots)