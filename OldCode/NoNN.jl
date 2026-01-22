```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Clean imports - no duplicates
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
using Lux

gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ModelsGeneralized.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")


"""
DEFINE INITIAL CONDITIONS
"""
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (5264, 7000)
datasize = 8
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 1.0
num_optimization_increments = 100

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"./input/trajectoryA_eccentric.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"./input/trajectoryB_eccentric.txt")
waveform_real_ecc = file2waveform(tsteps,"./input/waveform_real_eccentric.txt")

e₀ = 0.085   # optimized value is near 0.099
x₀ = 6.484530581221468
χ₀ = 3.66 # optimized value is near 3.66
p₀ = x₀ * (1+e₀*cos(χ₀)) / mass1
u0_pe_coordinates = Float32[χ₀, 0.0, 6.484530872637143*2+2.1, 0.16384305990318113] # χ₀, ϕ₀, p₀, e₀
p = u0_pe_coordinates[3]
e = u0_pe_coordinates[4]
E0, L0 = pe_2_EL(p, e)[2:3]
# r0 = p / (1 + e)  # Correct periapsis radius = 12.368763
r0 = 6.482964537732796*2
E0 = 0.8437969360351569
pᵣ0 = 0.01
L0 = 3.5061003277
u0 = [0, r0, pi/2, 0, E0, pᵣ0, 0, -L0, 0]

"""
DEFINE ODE MODEL FOR BINARY BLACK HOLE SYSTEM!
"""
function ODE_model_dual(du, u, p, t)
    du = GENERIC(du, u, model_params, t)
    return du
end

"""
SET UP ODE MODEL + SOLVE IT + CONVERT TO WAVEFORM
"""
prob_nn_dual = ODEProblem(ODE_model_dual, u0, tspan)
soln_nn_dual = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual, waveform_nn_imag_dual = compute_waveform(dt_data, soln_nn_dual, mass_ratio; coorbital=false)

# Build orbits in the format expected by h_22_strain_two_body: 2×N, rows = (x; y)
orbit1_true = [x_ecc'; y_ecc']
orbit2_true = [x2_ecc'; y2_ecc']

# Compute "truth" waveform using the SAME derivative/scaling as prediction
h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, orbit1_true, mass1, orbit2_true, mass2)

# Compare apples-to-apples with your predicted waveform

plot(tsteps, waveform_real_ecc, label = "True (original)")
plot!(tsteps, h_plus_true, label="True (used compute_waveform on true x(t), y(t))", legend=:topright)# plot!(tsteps, waveform_nn_real_dual, label="Predicted")
# plot!(tsteps, waveform_nn_real_dual, label = "ODE Model")