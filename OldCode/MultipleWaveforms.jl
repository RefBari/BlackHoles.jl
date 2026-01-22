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

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ModelsGeneralized.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

""" ************************
STEP 1: IMPORT TRAINING DATA
************************ """
tspan = (0, 1.8e3)
datasize = 100
tsteps = range(tspan[1], tspan[2], length = datasize) 

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p15_e0p5.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p15_e0p5.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p15_e0p5.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p15_e0p5.txt")

# Gather waveform data
x_ecc_2, y_ecc_2 = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p30_e0p5.txt")
x2_ecc_2, y2_ecc_2 = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p30_e0p5.txt")
waveform_real_ecc_2 = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p30_e0p5.txt")
waveform_imag_ecc_2 = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p30_e0p5.txt")

""" *******************************
STEP 2: DEFINE SIMULATION PARAMETERS
******************************* """
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

dt_data = tsteps[2] - tsteps[1]
dt = 1.0
num_optimization_increments = 10

""" *******************************
STEP 3: DEFINE INITIAL CONDITIONS
******************************* """
p = 15
e = 0.5

p_2 = 30
e_2 = 0.5

const rvals_penalty = range(10, 60; length = 100)

# E0, L0 = circular_pt_L(R)
E0_base, L0_base = eccentric_pt_L(p, e) # Newtonian IC
E0_base = -1 * E0_base
# E0_base_schwarzschild, L0_base_schwarzschild = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0 = p / (1 - e)
pᵣ0 = 0.0

# E0, L0 = circular_pt_L(R)
E0_base_2, L0_base_2 = eccentric_pt_L(p_2, e_2) # Newtonian IC
E0_base_2 = -1 * E0_base_2
# E0_base_2, L0_base_2 = pe_2_EL(p_2, e_2)[2:3] # Schwarzschild IC
r0_2 = p_2 / (1 - e_2)
pᵣ0_2 = 0.0

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
            layer.bias .= 0 # Force output near 0
        else  # Hidden layers
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
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
    conservative_features = [u[2]]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params,
    dE0 = 0.0, # Orbit 1 IC
    dL0 = 0.0,
    dE0_2 = 0.0, # Orbit 2 IC
    dL0_2 = 0.0
)

""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """
function make_u0(params)
    E0 = E0_base + params.dE0
    L0 = L0_base + params.dL0

    return [
        0.0, # t
        r0, # r
        pi/2, # θ
        0.0, # ϕ
        E0, # pₜ
        0.0, # pᵣ
        0.0, # p_θ
        L0, # p_ϕ
        0.0
    ]
end

function make_u0_2(params)
    E0_2 = E0_base_2 + params.dE0_2
    L0_2 = L0_base_2 + params.dL0_2

    return [
        0.0, # t
        r0_2, # r
        pi/2, # θ
        0.0, # ϕ
        E0_2, # pₜ
        0.0, # pᵣ
        0.0, # p_θ
        L0_2, # p_ϕ
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
************************************************************"""
u0_init = make_u0(NN_params)
prob_nn_dual = ODEProblem(ODE_model_dual, u0_init, tspan, NN_params)
soln_nn_dual = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual, waveform_nn_imag_dual = compute_waveform(dt_data, soln_nn_dual, mass_ratio; coorbital=false)
orbit = soln2orbit(soln_nn_dual)
pred_orbit1_init, pred_orbit2_init = one2two(orbit, 1, mass_ratio)

u0_init_2 = make_u0_2(NN_params)
prob_nn_dual_2 = ODEProblem(ODE_model_dual, u0_init_2, tspan, NN_params)
soln_nn_dual_2 = Array(solve(prob_nn_dual_2, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual_2, waveform_nn_imag_dual_2 = compute_waveform(dt_data, soln_nn_dual_2, mass_ratio; coorbital=false)
orbit_2 = soln2orbit(soln_nn_dual_2)
pred_orbit1_init_2, pred_orbit2_init_2 = one2two(orbit_2, 1, mass_ratio)

plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real Orbit 1")
plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Predicted Orbit 1")
plot!(x_ecc_2, y_ecc_2, aspect_ratio=:equal, linewidth = 2, label = "Real Orbit 2")
plot!(pred_orbit1_init_2[1,:], pred_orbit1_init_2[2,:], aspect_ratio=:equal, linewidth = 2, label = "Predicted Orbit 2")

plot(waveform_real_ecc, label = "Real Orbit 1")
plot!(waveform_nn_real_dual, label = "Predicted Orbit 1")
plot(waveform_imag_ecc, label = "Real Orbit 2")
plot!(waveform_nn_imag_dual, label = "Predicted Orbit 2")

plot(waveform_real_ecc_2, label = "Real")
plot!(waveform_nn_real_dual_2, label = "Prediction")
plot(waveform_imag_ecc_2, label = "Real")
plot!(waveform_nn_imag_dual_2, label = "Prediction")

""" ********************************************************
STEP 8.1: TEST INITIAL SOLUTION BEFORE ANY TRAINING
************************************************************"""
# Define an r range (just outside the Schwarzschild radius)
rvals_init = range(10, 60; length=datasize)

# Evaluate NN-predicted multiplicative corrections
f_tt_pred_init = []
f_rr_pred_init = []
f_pp_pred_init = []

for r in rvals_init
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    # push!(f_tt_pred_init, exp(out[1]))
    push!(f_rr_pred_init, out[1])
    push!(f_pp_pred_init, out[2])
end

f = 1 .- 2 ./ rvals_init

""" 8.1A: SCHWARZSCHILD METRIC COMPONENTS (INITIAL) """
# g_tt_true = -f^(-1)
g_rr_true_init = f
g_ϕϕ_true_init = 1 ./ rvals_init.^2

""" 8.1B: PREDICTED NN METRIC COMPONENTS (INITIAL) """
# g_tt_pred = ...
g_rr_pred_init = [((1 + 2/rvals_init)^(-1)) * (1 + f_rr) for (rvals_init, f_rr) in zip(rvals_init, f_rr_pred_init)]
g_ϕϕ_pred_init = [rvals_init^(-2) * (1 + 2/rvals_init)^(-1) * (1 + f_pp) for (rvals_init, f_pp) in zip(rvals_init, f_pp_pred_init)]

""" 8.1C: NEWTONIAN METRIC COMPONENTS (INITIAL) """
# g_tt_orig = ...
g_rr_orig_init = [((1 + 2/rvals_init)^(-1)) for (rvals_init, f_rr) in zip(rvals_init, f_rr_pred_init)]
g_ϕϕ_orig_init = [rvals_init^(-2) * (1 + 2/rvals_init)^(-1) for (rvals_init, f_pp) in zip(rvals_init, f_pp_pred_init)]

""" 8.1D: PLOT METRIC COMPONENTS (INITIAL) """

# plt1 = plot(rvals_init, g_tt_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
# plot!(plt1, rvals_init, g_tt_pred_init, lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{tt}", title=L"g^{tt}(r)")

plt2 = plot(rvals_init[1:30], g_rr_true_init[1:30], lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt2, rvals_init[1:30], g_rr_pred_init[1:30], lw=2, color=:red, label="Initial Prediction", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")
plot!(plt2, rvals_init[1:30], g_rr_orig_init[1:30], lw=2, label="Newtonian", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")

plt3 = plot(rvals_init, g_ϕϕ_true_init, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt3, rvals_init, g_ϕϕ_pred_init, lw=2, color=:red, label="Initial Prediction", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")
plot!(plt3, rvals_init, g_ϕϕ_orig_init, lw=2, label="Newtonian", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])

    # Build Initial Conditions from current params
    u0_local = make_u0(NN_params)
    u0_local_2 = make_u0_2(NN_params)

    # Redefine ODE Problem using these ICs and params
    prob = ODEProblem(ODE_model_dual, u0_local, tspan, NN_params)
    pred_soln = solve(prob, Tsit5();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    prob_2 = ODEProblem(ODE_model_dual, u0_local_2, tspan, NN_params)
    pred_soln_2 = solve(prob_2, Tsit5();
                      saveat = saveat, dt = dt, 
                      adaptive=false, verbose = false, 
                      sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real_2, pred_waveform_imag_2 = compute_waveform(dt_data, pred_soln_2, mass_ratio)

    N = length(pred_waveform_real)

    data_loss_real = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N]))
                    # + ( sum(abs2, waveform_real_ecc_2[1:N] .- pred_waveform_real_2)) / (sum(abs2, waveform_real_ecc_2[1:N]))
    data_loss_imag = ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))
                    # + ( sum(abs2, waveform_imag_ecc_2[1:N] .- pred_waveform_imag_2)) / (sum(abs2, waveform_imag_ecc_2[1:N]))
    tikhonov = 1e-8 * sum(abs2, NN_params)

    loss = data_loss_real + data_loss_imag + tikhonov 

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln, 
                 pred_waveform_real_2, pred_waveform_imag_2, pred_soln_2
end

# Test loss function
loss(NN_params)
losses = []

""" *************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """
optimization_increments = [1, 4, 6, 8, 10]
for i in optimization_increments
    println("optimization increment :: ", i, " of ", num_optimization_increments)
    opt_first = optimization_increments[1]         # 40
    opt_last  = optimization_increments[end]       # 100
    frac = 0.10 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]
    println("Training increment ", i, "/", num_optimization_increments,
        ": using ", length(tsteps_increment), " of ", length(tsteps),
        " points (", round(frac*100; digits=1), "% of total data)")


    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    
    function scalar_loss(p)
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln,
                  pred_waveform_real_2, pred_waveform_imag_2, pred_soln_2 = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
    
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln, 
                  pred_waveform_real_2, pred_waveform_imag_2, pred_soln_2 = tmp_loss(p)
    
        push!(losses, loss_val)
        display(loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Re[Gravitational] Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Re[Gravitational] Wave")    
        display(plt1)

        plt2 = plot(tsteps[startPoint:N], waveform_real_ecc_2[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave 2")
        plot!(plt2, tsteps[startPoint:N], pred_waveform_real_2[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave 2")    
        display(plt2)
        
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit_relative_2 = soln2orbit(pred_soln_2)

        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)
        pred_orbit1_2, pred_orbit2_2 = one2two(pred_orbit_relative_2, 1, mass_ratio)
  
        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="orbit data", camera=(35,20))
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN")
        plot!(plt3, x_ecc_2[startPoint:N], y_ecc_2[startPoint:N], lw=2, alpha=1, label="orbit data 2", camera=(35,20))
        plot!(plt3, pred_orbit1_2[1,startPoint:N], pred_orbit1_2[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN 2")
        display(plt3)
        
        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    res = nothing 

    if i < 5
        print("i:", i, ", lr: 3\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=7, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 50)
    elseif i == 5
        print("i:", i, ", lr: 2\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 40)
    elseif i == optimization_increments[end-3]  # 97
        print("i:", i, ", lr: 1\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=4, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 20)
    elseif i == optimization_increments[end-2]  # 98
        print("i:", i, ", lr: 1e-1\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=3, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 10)
    elseif i == optimization_increments[end-1]  # 99
        print("i:", i, ", lr: 1e-2\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), callback=opt_callback, allow_f_increases=false, iterations = 5)
    else  # 100'
        print("i:", i, ", lr: 1e-3\n")
        res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), callback=opt_callback, maxiters = 30, allow_f_increases=true, iterations = 1)
    end

    global NN_params = res.u
    print("The final optimized parameters are:", NN_params)
end

""" ********************************
STEP 11: VISUALIZE METRIC COMPONENTS
******************************** """

# Define an r range (just outside the Schwarzschild radius)
rvals = range(10, 60; length=100)

# Evaluate NN-predicted multiplicative corrections
# f_tt_pred = []
f_rr_pred = []
f_pp_pred = []

for r in rvals
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    # push!(f_tt_pred, exp(out[1]))
    push!(f_rr_pred, out[1])
    push!(f_pp_pred, out[2])
end

f = 1 .- 2 ./ rvals

""" SCHWARZSCHILD METRIC COMPONENTS """
# g_tt_true = -f^(-1)
g_rr_true = f
g_ϕϕ_true = 1 ./ rvals.^2

""" PREDICTED NN METRIC COMPONENTS """
# g_tt_pred = ...
g_rr_pred = [((1 + 2/r)^(-1)) * (1 + f_rr) for (r, f_rr) in zip(rvals, f_rr_pred)]
g_ϕϕ_pred = [r^(-2) * (1 + 2/r)^(-1) * (1 + f_pp) for (r, f_pp) in zip(rvals, f_pp_pred)]

""" NEWTONIAN METRIC COMPONENTS """
# g_tt_orig = ...
g_rr_orig = [((1 + 2/r)^(-1)) for (r, f_rr) in zip(rvals, f_rr_pred)]
g_ϕϕ_orig = [r^(-2) * (1 + 2/r)^(-1) for (r, f_pp) in zip(rvals, f_pp_pred)]

max_num = 100

plt2 = plot(rvals[1:max_num], g_rr_true[1:max_num], lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt2, rvals[1:max_num], g_rr_orig[1:max_num], lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")
plot!(plt2, rvals[1:max_num], g_rr_pred[1:max_num], lw=2, label="Predicted", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")
# display(plt2)

plt3 = plot(rvals[1:max_num], g_ϕϕ_true[1:max_num], lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt3, rvals[1:max_num], g_ϕϕ_orig[1:max_num], lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")
plot!(plt3, rvals[1:max_num], g_ϕϕ_pred[1:max_num], lw=2, label="Predicted", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")
# display(plt3)

u0_opt = make_u0(NN_params)
prob_opt = ODEProblem(ODE_model_dual, u0_opt, tspan, NN_params)
optimized_solution = solve(prob_opt, Tsit5(), saveat = tsteps, dt = dt, adaptive=false)

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)
pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

u0_opt_2 = make_u0_2(NN_params)
prob_opt_2 = ODEProblem(ODE_model_dual, u0_opt_2, tspan, NN_params)
optimized_solution_2 = solve(prob_opt_2, Tsit5(), saveat = tsteps, dt = dt, adaptive=false)
pred_waveform_real_2, pred_waveform_imag_2 = compute_waveform(dt_data, optimized_solution_2, mass_ratio)
pred_orbit_2 = soln2orbit(optimized_solution_2)
pred_orbit1_nn_2, pred_orbit2_nn_2 = one2two(pred_orbit_2, 1, 1)

plot(pred_waveform_real)
plot!(waveform_real_ecc)

plt4 = plot(x_ecc, y_ecc, linewidth = 2, label = "truth")
plot!(plt4, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")
plot(plt4, x_ecc_2[1,:], y_ecc_2[2,:], linewidth = 2, label = "prediction", title = "Orbits")
plot!(plt4, pred_orbit1_nn_2[1,:], pred_orbit1_nn_2[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")


plot!(plt4, x2_ecc, y2_ecc, linewidth = 2, label = "truth 2")
plot!(plt4, pred_orbit2_nn[1,:], pred_orbit2_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")

# layout = @layout [a b; c d]
# combined_plot = plot(plt1, plt2, plt3, plt4,
#                      layout = layout,
#                      size = (900, 700),
#                      margin = 5mm)
# display(combined_plot)

plt5 = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Relative Loss", legend = false)
display(plt5)

"""
STEP 12: TESTING THE RATIO HYPOTHESIS
"""

r = optimized_solution[2,:]
ϕ = optimized_solution[4,:]
pᵣ = optimized_solution[6,:]
p_ϕ = optimized_solution[8,:]

N = length(tsteps)

dr_dt = zeros(N)
dϕ_dt = zeros(N)

for i in 2:N-1
    print("\ndt is ", dt)
    dr_dt[i] = (r[i+1]-r[i-1]) / (2* (tsteps[2]-tsteps[1]))
    dϕ_dt[i] = (ϕ[i+1]-ϕ[i-1]) / (2* (tsteps[2]-tsteps[1]))
end

function g_rr_term(r, NN_params)
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    f_rr = out[1]
    return (1+(2/r))^(-1) * (1+f_rr)
end 

function g_ϕϕ_term(r, NN_params)
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    f_ϕϕ = out[2]
    return r^(-2) * (1+(2/r))^(-1) * (1 + f_ϕϕ)
end

function g_tt_term(r)
    return -(1-(2/r))^(-1)
end

E = E0_base + NN_params.dE0
L = L0_base + NN_params.dL0

pₜ = E
p_ϕ = -L

dr_dt_H = zeros(N)
dϕ_dt_H = zeros(N)

for i in 1:N
    gtt_term = g_tt_term(r[i])
    grr_term = g_rr_term(r[i], NN_params)
    gϕϕ_term = g_ϕϕ_term(r[i], NN_params)

    dr_dt_H[i] = (grr_term * pᵣ[i]) / (gtt_term * pₜ)
    dϕ_dt_H[i] = (gϕϕ_term * p_ϕ) / (gtt_term * pₜ)
end

plot(dr_dt_H, label = L"\frac{g^{rr}p_r}{g^{tt}p_t}")
plot!(dr_dt,label = L"\frac{dr}{dt}")

plot(dϕ_dt_H, label = L"\frac{g^{\phi\phi}p_\phi}{g^{tt}p_t}")
plot!(dϕ_dt, label = L"\frac{d\phi}{dt}")