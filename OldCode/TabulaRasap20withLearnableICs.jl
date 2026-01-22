cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using LaTeXStrings
using Measures
using SymbolicRegression
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
using Optim
using Optimization
using OptimizationOptimisers
using Plots
using Lux
using Zygote

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

gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/GaussianModel.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")


""" ************************
STEP 1: IMPORT TRAINING DATA
************************ """
tspan = (0, 5.0f3)
datasize = 300
tsteps = range(tspan[1], tspan[2], length = datasize) 

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p20_e0p5.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p20_e0p5.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p20_e0p5.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p20_e0p5.txt")

""" *******************************
STEP 2: DEFINE SIMULATION PARAMETERS
******************************* """
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

dt_data = tsteps[2] - tsteps[1]
dt = 10
num_optimization_increments = 20

""" *******************************
STEP 3: DEFINE INITIAL CONDITIONS
******************************* """
p = 20
e = 0.5

r_min = p / (1+e)
r_max = p / (1-e)

r0 = p / (1 - e)

""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """
#  # Neural network setup 
NN_Conservative = Chain(
    Dense(2, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 1),
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
        # Apply small random noise to ALL weights to break symmetry
        layer.weight .= 1e-4 * randn(rng, eltype(layer.weight), size(layer.weight))
        
        # Initialize biases to 0
        layer.bias .= 0
    end
end 

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)

""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """
function make_u0(params)
    # Explicit Relativistic Calculation for p=20, e=0.5
    # L_newtonian would be sqrt(20) = 4.47 (Too weak!)
    # L_relativistic is approx 4.88
    
    num = p^2
    den = p - 3 - e^2
    L_rel = sqrt(num / den) # L ≈ 4.887
    
    # E calculation
    num_E = (p - 2 - 2*e) * (p - 2 + 2*e)
    den_E = p * (p - 3 - e^2)
    E_rel = sqrt(num_E / den_E)

    # Add corrections
    E0 = E_rel
    L0 = L_rel
    
    r0 = p/(1 - e)

    return [0.0, r0, pi/2, 0.0, -E0, 0.0, 0.0, L0, 0.0]
end

function NN_adapter_dual(u, params)
    # Conservative network
    r = u[2]
    r_inv = 20 / r
    conservative_features = [r_inv, r_inv^2]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end

""" ********************************
STEP 7: CREATE FUNCTION FOR ODE MODEL
*********************************"""
function ODE_model(du, u, p, t)
    du = GENERIC_Newtonian(du, u, model_params, t)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
*********************************************************"""

print("Initial NN Params:", NN_params.conservative.layer_3)

initialConditions = make_u0(NN_params)
prob = ODEProblem(ODE_model, initialConditions, tspan, NN_params)
soln = solve(prob, RK4(), saveat = tsteps, dt = dt)
orbit_pred = soln2orbit(soln)
orbit1_pred, orbit2_pred = one2two(orbit_pred, 1, mass_ratio)
h₊_pred, hₓ_pred = compute_waveform(dt_data, soln, mass_ratio; coorbital=false)

maxIndex = 300

plot(x_ecc[1:maxIndex], y_ecc[1:maxIndex], aspect_ratio=:equal, linewidth = 1, label = "Real")
plot!(orbit1_pred[1,1:maxIndex], orbit1_pred[2,1:maxIndex], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc[1:maxIndex], label = "Real")
plot!(h₊_pred[1:maxIndex], label = "Prediction")
plot(waveform_imag_ecc, label = "Real")
plot!(hₓ_pred, label = "Prediction")

r_soln_min = r_min
r_soln_max = r_max

r_vals_initial_testing = range(r_soln_min, r_soln_max, length = 101)

grr_pred_initial = []
grr_newtonian_array = []
grr_schwarzschild_array = []

gϕϕ_pred_initial = []
gϕϕ_newtonian_array = []
gϕϕ_schwarzschild_array = []

""" g^rr Functions """
function grr_pred(r, NN_params)
    r_inv = 20 / r

    out, _ = NN_Conservative([r_inv, r_inv^2], NN_params.conservative, NN_Conservative_state)

    scale_rr = 0.10
    f_rr_NN_correction = 1.0 - scale_rr * sigmoid(out[1] - 1.3)

    open("New_p20_Results_Attempt22.txt", "a") do io
        println(io, "\nr:", r, " \n[ Raw gᵣᵣ Correction: ", out[1], "| gᵣᵣ Correction (after 1.0 - 0.1*sigmoid(out[1] - 1.3)): ", f_rr_NN_correction, " ]")
        println(io, "[ Current gᵣᵣ Value: ", (1+(2/r))^(-1) * (f_rr_NN_correction), "| True gᵣᵣ Value: ", 1 - (2/r), " ]\n")
    end    
    print("\nr:", r, " \n[ Raw gᵣᵣ Correction: ", out[1], "| gᵣᵣ Correction (after 1.0 - 0.1*sigmoid(out[1] - 1.3)): ", f_rr_NN_correction, " ]")
    print("\n[ Current gᵣᵣ Value: ", (1+(2/r))^(-1) * (f_rr_NN_correction), "| True gᵣᵣ Value: ", 1 - (2/r), " ]\n")
    return (1+(2/r))^(-1) * (f_rr_NN_correction), f_rr_NN_correction
end

function grr_newtonian(r)
    return (1+(2/r))^(-1)
end

function grr_schwarzschild(r)
    return (1-(2/r))
end

""" g^ϕϕ Functions """
function gϕϕ_pred(r, NN_params)
    r_inv = 20 / r

    out, _ = NN_Conservative([r_inv, r_inv^2], NN_params.conservative, NN_Conservative_state)

    scale_phi = 0.3
    f_ϕϕ_NN_correction = 1.0 + scale_phi * sigmoid(out[2])
    open("New_p20_Results_Attempt22.txt", "a") do io
        println(io, "\n[ Raw gϕϕ Correction: ", out[2], "| gϕϕ Correction (after 1.0 + 0.3 * sigmoid(out[2])): ", f_ϕϕ_NN_correction, " ]")
        println(io, "[ Current gϕϕ Value: ", r^(-2) * (1+(2/r))^(-1) * (f_ϕϕ_NN_correction), "| True gϕϕ Value: ", r^(-2), " ]\n")
    end    
    print("\n[ Raw gϕϕ Correction: ", out[2], "| gϕϕ Correction (after 1.0 + 0.3 * sigmoid(out[2])): ", f_ϕϕ_NN_correction, " ]\n")
    print("[ Current gϕϕ Value: ", r^(-2) * (1+(2/r))^(-1) * (f_ϕϕ_NN_correction), "| True gϕϕ Value: ", r^(-2), " ]\n")
    return r^(-2) * (1+(2/r))^(-1) * (f_ϕϕ_NN_correction), f_ϕϕ_NN_correction
end

function gϕϕ_newtonian(r)
    return r^(-2) * (1+(2/r))^(-1)
end

function gϕϕ_schwarzschild(r)
    return r^(-2)
end

""" Plot g^rr & g^ϕϕ Initial Prediction vs. Newtonian vs. Schwarzschild """

for i in range(1, length(r_vals_initial_testing))
    
    # print("g^rr Corrections (Should be ~0.01): ", grr_pred(r_vals_initial_testing[i], NN_params)[2], "\n")
    push!(grr_pred_initial, grr_pred(r_vals_initial_testing[i], NN_params)[1])
    push!(grr_newtonian_array, grr_newtonian(r_vals_initial_testing[i]))
    push!(grr_schwarzschild_array, grr_schwarzschild(r_vals_initial_testing[i]))

    # print("g^ϕϕ Corrections (Should be 0): ", gϕϕ_pred(r_vals_initial_testing[i], NN_params)[2], "\n")
    # push!(gϕϕ_pred_initial, gϕϕ_pred(r_vals_initial_testing[i], NN_params)[1])
    # push!(gϕϕ_newtonian_array, gϕϕ_newtonian(r_vals_initial_testing[i]))
    # push!(gϕϕ_schwarzschild_array, gϕϕ_schwarzschild(r_vals_initial_testing[i]))

end

initial_grr_pred_plot = plot(r_vals_initial_testing, grr_pred_initial, title = L"g^{rr}", label = "Initial g^rr Predicted", linewidth = 2)
plot!(r_vals_initial_testing, grr_newtonian_array, label = "Initial g^rr Newtonian", linewidth = 2)
plot!(r_vals_initial_testing, grr_schwarzschild_array, label = "Initial g^rr Schwarzschild", linewidth = 2)  

# initial_gϕϕ_pred_plot = plot(r_vals_initial_testing, gϕϕ_pred_initial, title = L"g^{ϕϕ}", label = "Initial g^ϕϕ Predicted", linewidth = 2)
# plot!(r_vals_initial_testing, gϕϕ_newtonian_array, label = "Initial g^ϕϕ Newtonian", linewidth = 2)
# plot!(r_vals_initial_testing, gϕϕ_schwarzschild_array, label = "Initial g^ϕϕ Schwarzschild", linewidth = 2)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """
# function loss(NN_params; saveat=tsteps)
#     # 1. Standard ODE Solve
#     tspan = (saveat[1], saveat[end])
#     u0_local = make_u0(NN_params)
#     prob = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    
#     # Solve
#     pred_soln = solve(prob, RK4();
#                       saveat = saveat, dt = dt, 
#                       adaptive=false, verbose = false, 
#                       sensealg=BacksolveAdjoint(checkpointing=true))
    
#     # 2. Standard Waveform Loss
#     pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio, coorbital=false)
#     N = length(pred_waveform_real)
#     N_safe = min(N, length(waveform_real_ecc)) - 5
#     valid_indices = 1:N_safe
    
#     data_loss_real = sum(abs2, waveform_real_ecc[valid_indices] .- pred_waveform_real[valid_indices]) / sum(abs2, waveform_real_ecc[valid_indices])
#     data_loss_imag = sum(abs2, waveform_imag_ecc[valid_indices] .- pred_waveform_imag[valid_indices]) / sum(abs2, waveform_imag_ecc[valid_indices])
    
#     # ---------------------------------------------------------
#     # 3. NEW: Physics-Informed Metric Constraint (g_tt * g_rr = -1)
#     # ---------------------------------------------------------
    
#     # # Extract r values from the trajectory (Row 2 is r)
#     # # Shape becomes (1, Number of Time Steps) which Lux expects
#     # r_batch = reshape(pred_soln[2, valid_indices], 1, :)

#     # r_vals = pred_soln[2, valid_indices]
#     # r_inv = 20.0 ./ r_vals
#     # # Stack them to create (2, N) matrix
#     # features_batch = vcat(reshape(r_inv, 1, :), reshape(r_inv.^2, 1, :))
    
#     # # Run the Neural Network on all r points simultaneously
#     # # We ignore the 'state' output for the loss calculation
#     # nn_out, _ = NN_Conservative(features_batch, NN_params.conservative, NN_Conservative_state)
    
#     # # Extract the raw output for g^rr (Row 1)
#     # raw_out_rr = nn_out[1, :]
    
#     # # Reconstruct the g^rr Correction Factor 
#     # # CRITICAL: This formula MUST match what is in 'GaussianModel.jl' exactly!
#     # scale_rr = 0.10
#     # bias_rr  = 1.3
#     # f_rr_correction = @. 1.0 - scale_rr * sigmoid(raw_out_rr - bias_rr)
    
#     # # Calculate the Predicted g^rr and Fixed g^tt
#     # # g^rr_pred = g^rr_Newtonian * correction
#     # g_rr_pred = @. ((1.0 + 2.0 ./ r_vals) ^ -1) * f_rr_correction
    
#     # # g^tt_fixed = -(1 - 2/r)^-1
#     # g_tt_fixed = @. -(1.0 - 2.0 ./ r_vals) ^ -1
    
#     # # The Constraint: g^tt * g^rr + 1 = 0
#     # # We perform element-wise multiplication and addition
#     # metric_violation = @. (g_tt_fixed * g_rr_pred) + 1.0
    
#     # # Compute the Mean Squared Error of the violation
#     # physics_loss = sum(abs2, metric_violation) / length(metric_violation)
    
#     # ---------------------------------------------------------
    
#     # 4. Combine Losses
#     # Give the physics loss a high weight (e.g., 1.0 or 10.0) because it is exact math.
#     # Tikhonov is usually small (1e-8).
#     tikhonov = 1e-8 * sum(abs2, NN_params)
    
#     total_loss = data_loss_real + data_loss_imag + tikhonov 

#     return total_loss, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params, waveform_real_ecc
# end

function loss(NN_params; saveat=tsteps)
    # 1. Standard ODE Solve
    tspan = (saveat[1], saveat[end])
    u0_local = make_u0(NN_params)
    prob = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    
    # Solve (Forward Pass)
    pred_soln = solve(prob, RK4();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    
    # 2. Waveform Loss (Keep it, but it will be swamped by the penalty)
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio, coorbital=false)
    N = length(pred_waveform_real)
    N_safe = min(N, length(waveform_real_ecc)) - 5
    valid_indices = 1:N_safe
    
    data_loss_real = sum(abs2, waveform_real_ecc[valid_indices] .- pred_waveform_real[valid_indices]) / sum(abs2, waveform_real_ecc[valid_indices])
    data_loss_imag = sum(abs2, waveform_imag_ecc[valid_indices] .- pred_waveform_imag[valid_indices]) / sum(abs2, waveform_imag_ecc[valid_indices])
    # ---------------------------------------------------------
    # 3. DIRECT METRIC SUPERVISION (The Sanity Check)
    # ---------------------------------------------------------
    
    # Get 1D vector of r values from the trajectory
    r_vals = pred_soln[2, valid_indices] 
    
    # A. Calculate True Targets (Schwarzschild)
    grr_true = @. 1.0 - 2.0 / r_vals
    gphi_true = @. 1.0 / (r_vals^2)
    
    # B. Calculate Predicted Metrics (Using NN)
    # -------------------------------------------------------
    # CRITICAL FIX: Create 2 inputs (u and u^2) to match the NN
    # -------------------------------------------------------
    u_val = 20.0 ./ r_vals
    # Stack u and u^2 to create a (2, N) matrix
    inv_r_batch = vcat(reshape(u_val, 1, :), reshape(u_val.^2, 1, :))
    
    nn_out, _ = NN_Conservative(inv_r_batch, NN_params.conservative, NN_Conservative_state)
    
    # -- Radial Component --
    # Match GENERIC_Newtonian Bias/Scale exactly!
    raw_out_rr = nn_out[1, :]
    scale_rr = 0.20
    bias_rr  = -1.1 # Match the last setting (+1.1 in GENERIC means -1.1 here)
    # Wait! Check your GENERIC_Newtonian code one last time. 
    # If GENERIC uses sigmoid(x + 1.1), here use sigmoid(x + 1.1).
    # Let's stick to the "+" convention for safety:
    f_rr_correction = @. 1.0 - scale_rr * sigmoid(raw_out_rr + 1.1)
    
    # Base Metric: Flat (1.0)
    g_rr_pred = @. 1.0 * f_rr_correction
    
    # -- Angular Component --
    raw_out_phi = nn_out[2, :]
    scale_phi = 0.10
    # If GENERIC uses sigmoid(x - 6.0), use -6.0 here.
    f_phi_correction = @. 1.0 + scale_phi * sigmoid(raw_out_phi - 6.0)
    
    # Base Metric: Flat (1/r^2)
    g_phi_pred = @. (1.0 / r_vals^2) * f_phi_correction
    
    # C. Calculate Metric Error (MSE)
    loss_grr = sum(abs2, g_rr_pred .- grr_true) / length(r_vals)
    loss_gphi = sum(abs2, g_phi_pred .- gphi_true) / length(r_vals)# ---------------------------------------------------------
    
    # 4. Total Loss with MASSIVE Penalty
    # The 1e6 weight forces the NN to fit the metric curves immediately.
    tikhonov = 1e-8 * sum(abs2, NN_params)
    
    total_loss = data_loss_real + data_loss_imag + 1e6 * (loss_grr + loss_gphi) + tikhonov 

    return total_loss, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params, waveform_real_ecc
end
# Test loss function
loss(NN_params)
losses = []

optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

print("********************************\nInitial Params:", NN_params.conservative.layer_3)
for i in optimization_increments
    println("------------ optimization increment : ", i, " of ", num_optimization_increments, " ------------")
    opt_first = optimization_increments[1]         # 1
    opt_last  = optimization_increments[end]       # 10
    frac = 0.05 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]
    println("Training increment ", i, "/", num_optimization_increments,
        ": using ", length(tsteps_increment), " of ", length(tsteps),
        " points (", round(frac*100; digits=1), "% of total data)")
    
    open("New_p20_Results_Attempt22.txt", "a") do io
        println(io, "------------ optimization increment : ", i, " of ", num_optimization_increments, " ------------")
        println(io, "Training increment ", i, "/", num_optimization_increments,
        ": using ", length(tsteps_increment), " of ", length(tsteps),
        " points (", round(frac*100; digits=1), "% of total data)")
    end    

    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    
    function scalar_loss(p)
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params, waveform_real_ecc = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
    
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params, waveform_real_ecc = tmp_loss(p)
    
        push!(losses, loss_val)
        print("\nLoss: ", loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        
        open("New_p20_Results_Attempt22.txt", "a") do io
            println(io, "\nLoss: ", loss_val)
            println(io, "\nTrue h_+:", waveform_real_ecc[startPoint:N], "\n")
            println(io, "Predicted h_+:", pred_waveform_real[startPoint:N], "\n")
        end

        print("\nTrue h_+:", waveform_real_ecc[startPoint:N], "\n")
        print("Predicted h_+:", pred_waveform_real[startPoint:N], "\n")
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave (p = 12, e = 0.5)")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave (p = 12, e = 0.5)")    
        display(plt1)

        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)

        grr_pred_training = []
        grr_newtonian_training = []
        grr_schwarzschild_training = []

        gϕϕ_pred_training = []
        gϕϕ_newtonian_training = []
        gϕϕ_schwarzschild_training = []

        r_vals_training_min = minimum(pred_soln[2, :])
        r_vals_training_max = maximum(pred_soln[2, :])
        r_vals_training = range(r_vals_training_min, r_vals_training_max, length = 101)

        for i in range(1, length(r_vals_training))
            # print("r:", r_vals_training[i], "\n")
            # print("g^rr Corrections (Should be 0): ", grr_pred(r_vals_training[i], NN_params)[2], "\n")
            push!(grr_pred_training, grr_pred(r_vals_training[i], NN_params)[1])
            push!(grr_newtonian_training, grr_newtonian(r_vals_training[i]))
            push!(grr_schwarzschild_training, grr_schwarzschild(r_vals_training[i]))

            push!(gϕϕ_pred_training, gϕϕ_pred(r_vals_training[i], NN_params)[1])
            push!(gϕϕ_newtonian_training, gϕϕ_newtonian(r_vals_training[i]))
            push!(gϕϕ_schwarzschild_training, gϕϕ_schwarzschild(r_vals_training[i]))

        end

        training_grr_pred_plot = plot(r_vals_training, grr_pred_training, title = L"g^{rr}", label = "Training g^rr Predicted", linewidth = 2)
        plot!(r_vals_training, grr_newtonian_training, label = "Training g^rr Newtonian", linewidth = 2)
        plot!(r_vals_training, grr_schwarzschild_training, label = "Training g^rr Schwarzschild", linewidth = 2)  

        training_gϕϕ_pred_plot = plot(r_vals_training, gϕϕ_pred_training, title = L"g^{ϕϕ}", label = "Training g^ϕϕ Predicted", linewidth = 2)
        plot!(r_vals_training, gϕϕ_newtonian_training, label = "Training g^ϕϕ Newtonian", linewidth = 2)
        plot!(r_vals_training, gϕϕ_schwarzschild_training, label = "Training g^ϕϕ Schwarzschild", linewidth = 2)
        
        combinedMetricPlots = plot(training_grr_pred_plot, training_gϕϕ_pred_plot, layout = (1,2))
        display(combinedMetricPlots)

        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit (p = 20, e = 0.5)", camera=(35,20))
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit (p = 12, e = 0.5)")
        display(plt3)

        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit (p = 20, e = 0.5)", camera=(35,20))
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit (p = 12, e = 0.5)")
        display(plt3)

        return false
    end

    # global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 10
        print("i:", i, ", lr: 1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 50)
    elseif i == 10
        print("i:", i, ", lr: 1e-1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 40)
    elseif i == optimization_increments[end-3]  # 97
        print("i:", i, ", lr: 1e-2\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 20)
    elseif i == optimization_increments[end-2]  # 98
        print("i:", i, ", lr: 1e-3\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-4, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 10)
    elseif i == optimization_increments[end-1]  # 99
        print("i:", i, ", lr: 1e-4\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-5, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 5)
    else  # 100'
        print("i:", i, ", lr: 1e-5\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-6, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 100, allow_f_increases=true, iterations = 1)
    end

    global NN_params = res.minimizer
end 


print("********************************\nFinal Params:", NN_params.conservative.layer_3)

u0_opt = make_u0(NN_params)
prob_opt = ODEProblem(ODE_model, u0_opt, tspan, NN_params)
optimized_solution = solve(prob_opt, RK4(), saveat = tsteps, dt = dt, adaptive=false)
pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)
pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)
plot(x_ecc[1:40], y_ecc[1:40], aspect_ratio=:equal, linewidth = 2, label = "Real Orbit")
plot!(pred_orbit1_nn[1,1:40], pred_orbit1_nn[2,1:40], aspect_ratio=:equal, linewidth = 2, label = "Optimized Prediction")
plot(waveform_real_ecc, label = "Real Waveform", linewidth = 2, title = L"h_{+}")
plot!(pred_waveform_real, label = "Optimized Prediction", linewidth = 2)

r_soln_min_final = minimum(optimized_solution[2,:])
r_soln_max_final = maximum(optimized_solution[2,:])

r_vals_final_testing = range(r_soln_min_final, r_soln_max_final, length = 101)

timespan = (0, 5f3)
datapoints = 300
tsteps = 2000/300

grr_pred_final = []
grr_newtonian_final = []
grr_schwarzschild_final = []

gϕϕ_pred_final = []
gϕϕ_newtonian_final = []
gϕϕ_schwarzschild_final = []

for i in range(1, length(r_vals_final_testing))
    print(r_vals_final_testing[i], "\n")

    # print("g^rr Corrections (Should be ~0.01): ", grr_pred(r_vals_final_testing[i], NN_params)[2], "\n")
    push!(grr_pred_final, grr_pred(r_vals_final_testing[i], NN_params)[1])
    push!(grr_newtonian_final, grr_newtonian(r_vals_final_testing[i]))
    push!(grr_schwarzschild_final, grr_schwarzschild(r_vals_final_testing[i]))

    # print("g^ϕϕ Corrections (Should be 0): ", gϕϕ_pred(r_vals_final_testing[i], NN_params)[2], "\n")
    push!(gϕϕ_pred_final, gϕϕ_pred(r_vals_final_testing[i], NN_params)[1])
    push!(gϕϕ_newtonian_final, gϕϕ_newtonian(r_vals_final_testing[i]))
    push!(gϕϕ_schwarzschild_final, gϕϕ_schwarzschild(r_vals_final_testing[i]))

end

initial_grr_pred_plot = plot(grr_pred_final, label = "Final g^rr Predicted", linewidth = 2, title = L"g^{rr}", xlabel = L"r")
plot!(grr_newtonian_final, label = "Final g^rr Newtonian", linewidth = 2, title = L"g^{rr}", xlabel = L"r")
plot!(grr_schwarzschild_final, label = "Final g^rr Schwarzschild", linewidth = 2)  

initial_gϕϕ_pred_plot = plot(gϕϕ_pred_final, label = "Final g^ϕϕ Predicted", linewidth = 2, title = L"g^{ϕϕ}", xlabel = L"r")
plot!(gϕϕ_newtonian_final, label = "Final g^ϕϕ Newtonian", linewidth = 2, title = L"g^{ϕϕ}", xlabel = L"r")
plot!(gϕϕ_schwarzschild_final, label = "Final g^ϕϕ Schwarzschild", linewidth = 2)  

plot(losses, xaxis=:log, yaxis=:log, ylabel = "Loss", xlabel = "Iteration", title = "Loss vs. Iteration", linewidth = 2)