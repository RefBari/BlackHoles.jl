```
    Inverse problem script to learn ODE model from SXS waveform data
```
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
x_ecc_p14, y_ecc_p14 = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p14_e0p5.txt")
x2_ecc_p14, y2_ecc_p14 = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p14_e0p5.txt")
waveform_real_ecc_p14 = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p14_e0p5.txt")
waveform_imag_ecc_p14 = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p14_e0p5.txt")

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
p = 12
e = 0.5

p_14= 14
e_14 = 0.5

r_min = p / (1+e)
r_max = p / (1-e)

r_min_p14 = p_14 / (1+e_14)
r_max_p14 = p_14 / (1-e_14)

# E0, L0 = circular_pt_L(R)
E0_base, L0_base = eccentric_pt_L(p, e) # Newtonian IC
# E0_base, L0_base = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0 = p / (1 - e)
pᵣ0 = 0.0


E0_base_p14, L0_base_p14 = eccentric_pt_L(p_14, e_14) # Newtonian IC
# E0_base, L0_base = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0_p14 = p_14 / (1 - e_14)

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
function make_u0(params)
    # max deviations allowed from Newtonian ICs
    ΔE = 1e-2    # E0 ∈ [E0_base - 0.02, E0_base + 0.02]
    ΔL = 0.5     # L0 ∈ [L0_base - 1.5, L0_base + 1.5]

    raw_E = params.dE0
    raw_L = params.dL0

    E0 = E0_base + ΔE * sigmoid(raw_E)   # subtract sigmoid ∈ (0,ΔE)
    L0 = L0_base + ΔL * sigmoid(raw_L)

    return [
        0.0, # t
        r0, # r
        pi/2, # θ
        0.0, # ϕ
        -E0, # pₜ
        0.0, # pᵣ
        0.0, # p_θ
        L0, # p_ϕ
        0.0
    ]
end

function make_u0_p14(params)
    # max deviations allowed from Newtonian ICs
    ΔE = 1e-2    # E0 ∈ [E0_base - 0.02, E0_base + 0.02]
    ΔL = 0.5     # L0 ∈ [L0_base - 1.5, L0_base + 1.5]

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

""" ********************************
STEP 7: CREATE FUNCTION FOR ODE MODEL
*********************************"""
function ODE_model_dual(du, u, p, t)
    du = GENERIC(du, u, model_params, t,
                                    NN=NN_adapter_dual, 
                                    NN_params=p)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
*********************************************************"""

print("Initial NN Params:", NN_params.conservative.layer_3)

u0_init = make_u0(NN_params)
prob_nn_dual = ODEProblem(ODE_model_dual, u0_init, tspan, NN_params)
soln_nn_dual = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual, waveform_nn_imag_dual = compute_waveform(dt_data, soln_nn_dual, mass_ratio; coorbital=false)
orbit = soln2orbit(soln_nn_dual)
pred_orbit1_init, pred_orbit2_init = one2two(orbit, 1, mass_ratio)

u0_init_p14 = make_u0_p14(NN_params)
prob_nn_dual_p14 = ODEProblem(ODE_model_dual, u0_init_p14, tspan, NN_params)
soln_nn_dual_p14 = Array(solve(prob_nn_dual_p14, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual_p14, waveform_nn_imag_dual_p14 = compute_waveform(dt_data, soln_nn_dual_p14, mass_ratio; coorbital=false)
orbit_p14 = soln2orbit(soln_nn_dual_p14)
pred_orbit1_init_p14, pred_orbit2_init_p14 = one2two(orbit_p14, 1, mass_ratio)

plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
plot!(pred_orbit1_init[1,1:5], pred_orbit1_init[2,1:5], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc, label = "Real")
plot!(waveform_nn_real_dual, label = "Prediction")
plot(waveform_imag_ecc, label = "Real")
plot!(waveform_nn_imag_dual, label = "Prediction")

plot(x_ecc_p14, y_ecc_p14, aspect_ratio=:equal, linewidth = 2, label = "Real")
plot!(pred_orbit1_init_p14[1,1:5], pred_orbit1_init_p14[2,1:5], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc_p14, label = "Real")
plot!(waveform_nn_real_dual_p14, label = "Prediction")
plot(waveform_imag_ecc_p14, label = "Real")
plot!(waveform_nn_imag_dual_p14, label = "Prediction")

""" ********************************************************
STEP 8.1: TEST INITIAL SOLUTION BEFORE ANY TRAINING
************************************************************"""

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
    out, _ = NN_Conservative([r, 1/r], NN_params.conservative, NN_Conservative_state)
    f_rr = -1 * (out[1])/r^2
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
    out, _ = NN_Conservative([r, 1/r], NN_params.conservative, NN_Conservative_state)
    f_ϕϕ = (out[2])
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
    
    print(r_vals_initial_testing[i], "\n")
    
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
    u0_local_p14 = make_u0_p14(NN_params)

    # Redefine ODE Problem using these ICs and params
    prob = ODEProblem(ODE_model_dual, u0_local, tspan, NN_params)
    pred_soln = solve(prob, Tsit5();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    prob_p14 = ODEProblem(ODE_model_dual, u0_local_p14, tspan, NN_params)
    pred_soln_p14 = solve(prob_p14, Tsit5();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real_p14, pred_waveform_imag_p14 = compute_waveform(dt_data, pred_soln_p14, mass_ratio)

    N = length(pred_waveform_real)

    data_loss_real = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N]))
                    +( sum(abs2, waveform_real_ecc_p14[1:N] .- pred_waveform_real_p14)) / (sum(abs2, waveform_real_ecc_p14[1:N]))
    data_loss_imag = ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))
                    +( sum(abs2, waveform_imag_ecc_p14[1:N] .- pred_waveform_imag_p14)) / (sum(abs2, waveform_imag_ecc_p14[1:N]))
    tikhonov = 1e-8 * sum(abs2, NN_params)

    loss = data_loss_real + data_loss_imag  + tikhonov 

    # print("Data_loss_real: ", data_loss_real, "Data_loss_imag: ", data_loss_imag, " Reg_loss: ", tikhonov, " Metric Asymptote: ", metric_asymptote, "\n")

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln, NN_params
end

# Test loss function
loss(NN_params)
losses = []

""" ************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """
optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("********************************\nInitial Params:", NN_params.conservative.layer_3)
for i in optimization_increments
    println("optimization increment :: ", i, " of ", num_optimization_increments)
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
        display(loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave")    
        display(plt1)

        grr_pred_training = []
        grr_newtonian_training = []
        grr_schwarzschild_training = []

        r_vals_training_min = minimum(pred_soln[2, :])
        r_vals_training_max = maximum(pred_soln[2, :])
        r_vals_training = range(r_vals_training_min, r_vals_training_max, length = 101)

        for i in range(1, length(r_vals_training))
            
            # print("r:", r_vals_training[i], "\n")
            
            # print("g^rr Corrections (Should be 0): ", grr_pred(r_vals_training[i], NN_params)[2], "\n")
            push!(grr_pred_training, grr_pred(r_vals_training[i], NN_params)[1])
            push!(grr_newtonian_training, grr_newtonian(r_vals_training[i]))
            push!(grr_schwarzschild_training, grr_schwarzschild(r_vals_training[i]))

        end

        training_grr_pred_plot = plot(r_vals_training, grr_pred_training, title = L"g^{rr}", label = "Training g^rr Predicted", linewidth = 2)
        plot!(r_vals_training, grr_newtonian_training, label = "Training g^rr Newtonian", linewidth = 2)
        plot!(r_vals_training, grr_schwarzschild_training, label = "Training g^rr Schwarzschild", linewidth = 2)  
        display(training_grr_pred_plot)

        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)

        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N],
        lw=2, alpha=1, label="orbit data",
        camera=(35,20))
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N],
      lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN")
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
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=10, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 50)
    elseif i == 10
        print("i:", i, ", lr: 1e-1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=7, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 40)
    elseif i == optimization_increments[end-3]  # 97
        print("i:", i, ", lr: 1e-2\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 20)
    elseif i == optimization_increments[end-2]  # 98
        print("i:", i, ", lr: 1e-3\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 10)
    elseif i == optimization_increments[end-1]  # 99
        print("i:", i, ", lr: 1e-4\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 5)
    else  # 100'
        print("i:", i, ", lr: 1e-5\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 30, allow_f_increases=true, iterations = 1)
    end

    global NN_params = res.minimizer
    print("The final optimized parameters are:", NN_params.conservative.layer_3,
          " | dE0: ", 1e-2 * sigmoid(NN_params.dE0), " | dL0: ", 0.05 * sigmoid(NN_params.dL0))
end

""" ********************************
STEP 11: VISUALIZE METRIC COMPONENTS
******************************** """

print("********************************\nFinal Params:", NN_params.conservative.layer_3)

u0_opt = make_u0(NN_params)
prob_opt = ODEProblem(ODE_model_dual, u0_opt, tspan, NN_params)
optimized_solution = solve(prob_opt, Tsit5(), saveat = tsteps, dt = dt, adaptive=false)
pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)
pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

r_soln_min_final = minimum(optimized_solution[2,:])
r_soln_max_final = maximum(optimized_solution[2,:])

r_vals_final_testing = range(r_soln_min_final, r_soln_max_final, length = 101)

timespan = (0, 2e3)
datapoints = 100
tsteps = 2000/100

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

initial_grr_pred_plot = plot(grr_pred_final, label = "Final g^rr Predicted", linewidth = 1)
plot!(grr_newtonian_final, label = "Final g^rr Newtonian", linewidth = 1)
plot!(grr_schwarzschild_final, label = "Final g^rr Schwarzschild", linewidth = 1)  

initial_gϕϕ_pred_plot = plot(gϕϕ_pred_final, label = "Final g^ϕϕ Predicted", linewidth = 1)
plot!(gϕϕ_newtonian_final, label = "Final g^ϕϕ Newtonian", linewidth = 1)
plot!(gϕϕ_schwarzschild_final, label = "Final g^ϕϕ Schwarzschild", linewidth = 1)  

plot(losses, xaxis=:log, yaxis=:log, ylabel = "Loss", xlabel = "Iteration", title = "Loss vs. Iteration", linewidth = 2)