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

gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/GaussianModel.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

""" ************************
STEP 1: IMPORT TRAINING DATA
************************ """

tspan = (0, 1.9999e3)
datasize = 100
tsteps = range(tspan[1], tspan[2], length = datasize+1) 

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p18_e0p1.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p18_e0p1.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p18_e0p1.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p18_e0p1.txt")

""" *******************************
STEP 2: DEFINE SIMULATION PARAMETERS
******************************* """
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

dt_data = tsteps[2] - tsteps[1]
dt = 1
num_optimization_increments = 10

""" *******************************
STEP 3: DEFINE INITIAL CONDITIONS
******************************* """
p = 18
e = 0.1

r_min = p / (1+e)
r_max = p / (1-e)

r0 = p / (1 - e)

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
            layer.bias[1] = 0 # Force output near 0
            layer.bias[2] = 0 # Force output near 0
        else  # Hidden layers
            layer.weight .= 0
            layer.bias .= 0
        end
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
    E0, L0 = pe_2_EL(p, e)[2:3]
    r0 = p/(1 - e)
    # print("\nThe predicted E & L are: ", E0, " and ", L0)
    return [
        0.0,    # t
        r0,     # r
        pi/2,   # θ
        0.0,    # ϕ
        -E0,    # p_t
        0.0,    # p_r
        0.0,    # p_θ
        L0,     # p_ϕ
        0.0
    ]
end

function NN_adapter_dual(u, params)
    # Conservative network
    conservative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
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
    du = GENERIC_Schwarzschild(du, u, model_params, t,
                                    NN=NN_adapter_dual, 
                                    NN_params=p)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
*********************************************************"""

print("Initial NN Params:", NN_params.conservative.layer_3)

initialConditions = make_u0(NN_params)
prob = ODEProblem(ODE_model, initialConditions, tspan, NN_params)
soln = solve(prob, Tsit5(), saveat = tsteps, dt = dt_data)
orbit_pred = soln2orbit(soln)
orbit1_pred, orbit2_pred = one2two(orbit_pred, 1, mass_ratio)
h₊_pred, hₓ_pred = compute_waveform(dt_data, soln, mass_ratio; coorbital=false)

plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
plot!(orbit1_pred[1,:], orbit1_pred[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc, label = "Real")
plot!(h₊_pred, label = "Prediction")
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
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    f_rr = -1e-1*sigmoid(out[1])
    open("output.txt", "w") do io
        println(io, "\nr:", r, " \n[ Raw gᵣᵣ Correction: ", out[1], "| gᵣᵣ Correction (after 1e-1 * sigmoid(out[1])):", f_rr, " ]")
        println(io, "\n[ Current gᵣᵣ Value: ", (1+(2/r))^(-1) * (1 + f_rr), "| True gᵣᵣ Value: ", 1 - (2/r), " ]\n")
    end    
    print("\nr:", r, " \n[ Raw gᵣᵣ Correction: ", out[1], "| gᵣᵣ Correction (after 1e-1 * sigmoid(out[1])):", f_rr, " ]")
    print("\n[ Current gᵣᵣ Value: ", (1+(2/r))^(-1) * (1 + f_rr), "| True gᵣᵣ Value: ", 1 - (2/r), " ]\n")
    return (1+(2/r))^(-1) * (1 + f_rr), f_rr
end

function grr_newtonian(r)
    return (1+(2/r))^(-1)
end

function grr_schwarzschild(r)
    return (1-(2/r))
end

""" g^ϕϕ Functions """
function gϕϕ_pred(r, NN_params)
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    f_ϕϕ = 1e-1*sigmoid(out[2])
    open("TabulaRasa_Results_p18.txt", "w") do io
        println(io, "\n[ Raw gϕϕ Correction: ", out[2], "| gϕϕ Correction (after 1e-1 * sigmoid(out[2])): ", f_ϕϕ, " ]\n")
        println(io, "\n[ Current gϕϕ Value: ", r^(-2) * (1+(2/r))^(-1) * (1 + f_ϕϕ), "| True gϕϕ Value: ", r^(-2), " ]\n")
    end    
    print("\n[ Raw gϕϕ Correction: ", out[2], "| gϕϕ Correction (after 1e-1 * sigmoid(out[2])): ", f_ϕϕ, " ]\n")
    print("\n[ Current gϕϕ Value: ", r^(-2) * (1+(2/r))^(-1) * (1 + f_ϕϕ), "| True gϕϕ Value: ", r^(-2), " ]\n")
    return r^(-2) * (1+(2/r))^(-1) * (1 + f_ϕϕ), f_ϕϕ
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
    push!(gϕϕ_pred_initial, gϕϕ_pred(r_vals_initial_testing[i], NN_params)[1])
    push!(gϕϕ_newtonian_array, gϕϕ_newtonian(r_vals_initial_testing[i]))
    push!(gϕϕ_schwarzschild_array, gϕϕ_schwarzschild(r_vals_initial_testing[i]))

end

initial_grr_pred_plot = plot(r_vals_initial_testing, grr_pred_initial, title = L"g^{rr}", label = "Initial g^rr Predicted", linewidth = 2)
plot!(r_vals_initial_testing, grr_newtonian_array, label = "Initial g^rr Newtonian", linewidth = 2)
plot!(r_vals_initial_testing, grr_schwarzschild_array, label = "Initial g^rr Schwarzschild", linewidth = 2)  

initial_gϕϕ_pred_plot = plot(r_vals_initial_testing, gϕϕ_pred_initial, title = L"g^{ϕϕ}", label = "Initial g^ϕϕ Predicted", linewidth = 2)
plot!(r_vals_initial_testing, gϕϕ_newtonian_array, label = "Initial g^ϕϕ Newtonian", linewidth = 2)
plot!(r_vals_initial_testing, gϕϕ_schwarzschild_array, label = "Initial g^ϕϕ Schwarzschild", linewidth = 2)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])

    # Build Initial Conditions from current params
    u0_local = make_u0(NN_params)

    # Redefine ODE Problem using these ICs and params
    prob = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = solve(prob, Tsit5();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio, coorbital=false)

    N = length(pred_waveform_real)

    data_loss_real = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N]))
    data_loss_imag = ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))
    tikhonov = 1e-8 * sum(abs2, NN_params)

    loss = data_loss_real + data_loss_imag  + tikhonov 

    # print("Data_loss_real: ", data_loss_real, "Data_loss_imag: ", data_loss_imag, " Reg_loss: ", tikhonov, " Metric Asymptote: ", metric_asymptote, "\n")

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params
end

# Test loss function
loss(NN_params)
losses = []

optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("********************************\nInitial Params:", NN_params.conservative.layer_3)
for i in optimization_increments
    println("------------ optimization increment :: ", i, " of ", num_optimization_increments, " ------------")
    opt_first = optimization_increments[1]         # 1
    opt_last  = optimization_increments[end]       # 10
    frac = 0.05 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]
    println("Training increment ", i, "/", num_optimization_increments,
        ": using ", length(tsteps_increment), " of ", length(tsteps),
        " points (", round(frac*100; digits=1), "% of total data)")


    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    
    function scalar_loss(p)
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
    
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params = tmp_loss(p)
    
        push!(losses, loss_val)
        print("\nLoss: ", loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        
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
        
        combinedMetricPlots = plot(training_grr_pred_plot, training_gϕϕ_pred_plot, layout = (1,2), size = (900,400))
        display(combinedMetricPlots)

        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit (p = 12, e = 0.5)", camera=(35,20))
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit (p = 12, e = 0.5)")
        display(plt3)

        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 10
        print("i:", i, ", lr: 1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=7, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 50)
    elseif i == 10
        print("i:", i, ", lr: 1e-1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 40)
    elseif i == optimization_increments[end-3]  # 97
        print("i:", i, ", lr: 1e-2\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=4, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 20)
    elseif i == optimization_increments[end-2]  # 98
        print("i:", i, ", lr: 1e-3\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 10)
    elseif i == optimization_increments[end-1]  # 99
        print("i:", i, ", lr: 1e-4\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 5)
    else  # 100'
        print("i:", i, ", lr: 1e-5\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 30, allow_f_increases=true, iterations = 1)
    end

    global NN_params = res.minimizer
end 