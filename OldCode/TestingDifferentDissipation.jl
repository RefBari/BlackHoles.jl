```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
# using DifferentialEquations
# using DiffEqFlux
using LinearAlgebra
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Lux
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
# using DiffEqSensitivity
using ComponentArrays
using Optimization
using OptimizationOptimisers
using OrdinaryDiffEq
gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/myModelsMan.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity
    
    M = 1
    E = sqrt( (p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)) )
    L = sqrt( (p^2 * M^2) / (p-3-e^2) )
  
    return [M, E, L]
  end

## Define the experiment parameters
u0_pe_coordinates = Float32[0.0, 0.0, 12.37, 1.0f-4]  # χ₀ = 0, φ₀ = 0 (periapsis)
p = u0_pe_coordinates[3]
e = u0_pe_coordinates[4]
E0, L0 = pe_2_EL(p, e)[2:3]
r0 = p / (1 + e)  # Correct periapsis radius = 12.368763

u0 = Float32[0, r0, pi/2, 0, E0, 0, 0, -L0, 0]

tspan = (5.20913f3, 6.78f3) # why this time span, specifically?
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100
mass_ratio = 1.0
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = mass_ratio/(1.0+mass_ratio)
mass2 = 1.0/(1.0+mass_ratio)

# # Gather waveform data
x, y = file2trajectory(tsteps,"./input/trajectoryA.txt")
waveform_real = file2waveform(tsteps,"./input/waveform_real.txt") 

plt = plot(tsteps, waveform_real,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data (Re)")

# # Update ODE_model to use ComponentArray and adapters
# function ODE_model_simple(du, u, p, t)
#     du = AbstractNROrbitModel(du, u, model_params, t)  # No NN arguments
#     return du
# end

# prob_simple = ODEProblem(ODE_model_simple, u0, tspan, nothing)
# soln_simple = solve(prob_simple, RK4(), saveat = tsteps, dt = dt)
# waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_simple, 1.0; coorbital=false)

# n_actual = length(soln_simple[2,:])
# tsteps_actual = tsteps[1:n_actual]

# plot!(plt, tsteps_actual, waveform_nn_real[1:n_actual],
#            markershape=:circle, markeralpha = 0.25,
#            linewidth = 2, alpha = 0.5,
#            label="waveform (No NN)")
# display(plt)

# pred_orbit = soln2orbit(soln_simple)
# pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

# x, y = file2trajectory(tsteps,"./input/trajectoryA.txt")

# # body 1 and body 2 orbits in the orbital plane
# plt2 = plot(title="Predicted orbits", aspect_ratio=:equal, legend=:topright)
# plot!(plt2, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linewidth=2, label="manual dissipation", alpha = 1, color = "orange")
# plot!(plt2, x, y, linewidth = 2, alpha = 0.5,
#            label="true circular inspiral", color = "blue")
# # plot!(plt2, pred_orbit2_nn[1,:], pred_orbit2_nn[2,:], linewidth=2, label="body 2")
