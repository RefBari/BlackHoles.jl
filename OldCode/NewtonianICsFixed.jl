cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using LaTeXStrings
using Measures
using SymbolicRegression
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
using Plots
using Lux
using Zygote

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/GaussianModel.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (0, 2e3)
datasize = 100
tsteps = range(tspan[1], tspan[2], length = datasize+1) 
dt_data = tsteps[2] - tsteps[1]
dt = 1
num_optimization_increments = 100

function ODE_model_Newtonian(du, u, p, t)
    du = GENERIC_Newtonian(du, u, model_params, t)
    return du
end

function ODE_model_Schwarzschild(du, u, p, t)
    du = GENERIC_Schwarzschild(du, u, model_params, t)
    return du
end

p = 20
e = 0.1

R = p / (1-e)

E0_newtonian, L0_newtonian = pe_2_EL_newtonian(p, e)
E0_schwarzschild, L0_schwarzschild = pe_2_EL(p, e)[2:3]

u0_newtonian = [0.0, R, pi/2, 0.0, -E0_newtonian, 0.0, 0.0, L0_newtonian, 0.0]
u0_schwarzschild = [0.0, R, pi/2, 0.0, -E0_schwarzschild, 0.0, 0.0, L0_schwarzschild, 0.0]

prob_schwarzschild = ODEProblem(ODE_model_Schwarzschild, u0_schwarzschild, tspan)
soln_schwarzschild = Array(solve(prob_schwarzschild, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
orbit_schwarzschild = soln2orbit(soln_schwarzschild)
blackHole_r1_sch, blackHole_r2_sch = one2two(orbit_schwarzschild, mass1, mass2)
h_plus_true_sch, h_cross_true_sch = h_22_strain_two_body(dt_data, blackHole_r1_sch, mass1, blackHole_r2_sch, mass2)
plot(orbit_schwarzschild[1,:], orbit_schwarzschild[2,:], linewidth = 2)
plot(h_plus_true_sch[1:13])

prob_newton = ODEProblem(ODE_model_Newtonian, u0_newtonian, tspan)
soln_newton = Array(solve(prob_newton, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
orbit_newton = soln2orbit(soln_newton)
blackHole_r1_newt, blackHole_r2_newt = one2two(orbit_newton, mass1, mass2)
h_plus_true_newt, h_cross_true_newt = h_22_strain_two_body(dt_data, blackHole_r1_newt, mass1, blackHole_r2_newt, mass2)
plot!(orbit_newton[1,1:18], orbit_newton[2,1:18], linewidth = 2)
plot!(h_plus_true_newt[1:13])

""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """
#  # Neural network setup 
NN_Conservative = Chain(
    Dense(1, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 2),
)

NN_Dissipative = Chain(
    Dense(1, 1, tanh), # Input: r
    Dense(1, 1, tanh),
    Dense(1, 1),
)

""" *******************************
STEP 5: INITIALIZE NEURAL NETWORKS
******************************* """
# Initialize parameters for both NNs
rng = MersenneTwister(222)
NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)
NN_Dissipative_params, NN_Dissipative_state = Lux.setup(rng, NN_Dissipative)

# Convert to appropriate precision
precision = Float64
NN_Conservative_params = Lux.fmap(x -> precision.(x), NN_Conservative_params)
NN_Dissipative_params = Lux.fmap(x -> precision.(x), NN_Dissipative_params)

for (i, layer) in enumerate(NN_Conservative_params)
    if ~isempty(layer)
        if i == length(NN_Conservative_params)  # Final layer
            layer.weight .= 0
            layer.bias .= 4 # Force output near 0
        else  # Hidden layers
            layer.weight .= 0
            layer.bias .= 0
        end
    end
end 

for (i, layer) in enumerate(NN_Dissipative_params)
    if ~isempty(layer)
        if i == length(NN_Dissipative_params)  # Final layer
            layer.weight .= 0.01 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.0  # Force output near 0
        else  # Hidden layers
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
        end
    end
end 

""" ******************************************************************
STEP 6: ASSIGN NEURAL NETWORK INPUTS & EXTRACT NEURAL NETWORK OUTPUTS
****************************************************************** """
# Now, create adapter functions that match the calling pattern expected by AbstractNROrbitModel
function NN_adapter_dual(u, params)
    # Conservative network
    conservative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params,
    dE0 = 0.0,
    dL0 = 0.0
)

""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """
function make_u0_p14(params)
    # max deviations allowed from Newtonian ICs
    ΔE = 0    # E0 ∈ [E0_base - 0.02, E0_base + 0.02]
    ΔL = 0     # L0 ∈ [L0_base - 1.5, L0_base + 1.5]

    raw_E = params.dE0
    raw_L = params.dL0

    E0 = E0_base_p14 + ΔE * sigmoid(raw_E)   # subtract sigmoid ∈ (0,ΔE)
    L0 = L0_base_p14 + ΔL * sigmoid(raw_L)

    return [
        0.0, # t
        r0_p14, # r
        pi/2, # θ
        0.0, # ϕ
        -E0, # pₜ
        0.0, # pᵣ
        0.0, # p_θ
        L0, # p_ϕ
        0.0
    ]
end
