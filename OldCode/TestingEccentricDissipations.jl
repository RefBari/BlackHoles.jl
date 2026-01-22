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


# # Gather waveform data

t₀ = 5263.7607
# χ₀ = 0 # optimized value is near 3.66
# p₀ = x₀ * (1+e₀*cos(χ₀))
# φ0 = atan(y[1], x[1])
# u0_pe_coordinates = Float32[χ₀, φ0, p₀, e₀] # χ₀, ϕ₀, p₀, e₀

tspan = (t₀, 6.769f3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100
mass_ratio = 1.0
model_params = [mass_ratio]

x_ecc, y_ecc = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")
waveform_ecc_real = file2waveform(tsteps,"./input/waveform_real_eccentric.txt")
waveform_ecc_imag = file2waveform(tsteps,"./input/waveform_imag_eccentric.txt")

u0_pe_coordinates = Float32[0.0, 0.0, 6.484530872637143*2+2.1, 0.16384305990318113]  # χ₀ = 0, φ₀ = 0 (periapsis)
p = u0_pe_coordinates[3]
e = u0_pe_coordinates[4]
E0_og, L0_og = pe_2_EL(p, e)[2:3]
r0_og = p / (1 + e)  # Correct periapsis radius = 12.368763

r0 = 12.9690618979
E0 = 0.8437969360351569
pᵣ0 = 0.01
L0 = 3.5061003277
u0 = [0, r0, pi/2, 0, E0, pᵣ0, 0, -L0, 0]

# # # Update ODE_model to use ComponentArray and adapters
function ODE_model_simple(du, u, p, t)
    du = AbstractNROrbitModel(du, u, model_params, t)  # No NN arguments
    return du
end

prob_simple = ODEProblem(ODE_model_simple, u0, tspan, nothing)
soln_simple = solve(prob_simple, RK4(), saveat = tsteps, dt = dt)
waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_simple, 1.0; coorbital=false)

maxSteps = 1000
n_actual = maxSteps
# n_actual = length(soln_simple[2,:])
tsteps_actual = tsteps[1:n_actual]

plt = plot(tsteps_actual, waveform_ecc_real[1:n_actual],
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform data (Re)")

plot!(plt, tsteps_actual, waveform_nn_real[1:n_actual],
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label="waveform (No NN)")
display(plt)

pred_orbit = soln2orbit(soln_simple)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

# body 1 and body 2 orbits in the orbital plane


plt2 = plot(title="Checking Initial Conditions", legend=:topright, aspect_ratio=:equal)
plot!(plt2, pred_orbit1_nn[1,1:maxSteps], pred_orbit1_nn[2,1:maxSteps], linewidth=2, label="manual dissipation", alpha = 1, color = "orange")
plot!(plt2, x_ecc[1:maxSteps], y_ecc[1:maxSteps], linewidth = 2, alpha = 1, label="true eccentric inspiral", color = "blue")

scatter!([pred_orbit1_nn[1,1]],[pred_orbit1_nn[2,1]],markersize=5, markercolor=:orange, label = "Guessed Start Point")
# scatter!([x_ecc[1,1]],[y_ecc[2,1]], markersize=5, markercolor=:blue, label = "True Start Point")
