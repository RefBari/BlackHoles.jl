```

    Inverse problem script to learn ODE model from SXS waveform data

```

include("TrainingDataFunctions.jl")
include("NNFunctions.jl")
include("Orbit2Waveform.jl")

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
gr()

""" ************************
STEP 1: IMPORT TRAINING DATA
************************ """
tspan = (0, 2e3)
datasize = 500
tsteps = range(tspan[1], tspan[2], length = datasize) 

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p20_e0.5.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p20_e0.5.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p20_e0.5.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p20_e0.5.txt")

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
p = 20
e = 0.5

r_min = p / (1+e)
r_max = p / (1-e)

E0_base, L0_base = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0 = p / (1 - e)

""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """
#  # Neural network setup 
NN_Conservative = Chain(
    Dense(1, 10, tanh),
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

for (i, layer) in enumerate(NN_Conservative_params)
    if ~isempty(layer)
        if i == length(NN_Conservative_params)  # Final layer
            layer.weight .= 1e-2 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= -7
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
function NN_adapter(u, params)
    scale_factor = 50

    # Conservative network
    conservative_features = [u[2] / scale_factor]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2]]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
    dissipative_output, _ = NN_Dissipative(dissipative_features, params.dissipative, NN_Dissipative_state)

    return (conservative = conservative_output, dissipative = dissipative_output)
end

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)

""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """
function make_u0(params)
    E0 = E0_base
    L0 = L0_base

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

""" ********************************
STEP 7: CREATE FUNCTION FOR ODE MODEL
*********************************"""
function ODE_model(du, u, p, t)
    du = Newtonian(du, u, model_params, t,
                                    NN=NN_adapter, 
                                    NN_params=p)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
*********************************************************"""
u0_init = make_u0(NN_params)
prob_nn = ODEProblem(ODE_model, u0_init, tspan, NN_params)
soln_nn = Array(solve(prob_nn, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_nn, mass_ratio; coorbital=false)
orbit = soln2orbit(soln_nn)
pred_orbit1_init, pred_orbit2_init = one2two(orbit, 1, mass_ratio)

plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc, label = "Real", linewidth=2)
plot!(waveform_nn_real, label = "Prediction", linewidth=2)
plot(waveform_imag_ecc, label = "Real", linewidth=2)
plot!(waveform_nn_imag, label = "Prediction", linewidth=2)
initial_metric_plot1, initial_metric_plot2 = plot_metric_components(NN_params, r_min, r_max)
display(initial_metric_plot1)
display(initial_metric_plot2)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])

    u0_local = make_u0(NN_params)
    prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = solve(prob_pred, Tsit5(); saveat = saveat, dt = dt, adaptive=false, verbose = false)
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    N = length(pred_waveform_real)
    waveform_loss = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N])) +
                    ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))
    tikhonov_reg = 1e-8 * sum(abs2, NN_params)
    loss = waveform_loss + tikhonov_reg

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(NN_params)
losses = []

""" *************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """

optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
progress_bar = Progress(length(optimization_increments);
                        dt = 1.0, color=:green, desc = "Optimization Progress", 
                        barlen = 40)

# print("********************************\nInitial Params:", NN_params.conservative.layer_3)
for i in optimization_increments
    println("\noptimization increment :: ", i, " of ", num_optimization_increments)
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
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
        # print("\n<------- NN_Conservative layer 3 weights and biases ------->:", p.conservative.layer_3)
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln = tmp_loss(p)
    
        push!(losses, loss_val)
        print("\nIteration $(length(losses)) | Loss: $(round(loss_val, digits = 5))")
    
        N = length(pred_waveform_real)
        startPoint = 1
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave",
                    xaxis = "Time", ylabel = L"h(t)", title = "Gravitational Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave",
              xaxis = "Time", ylabel = L"h(t)", title = "Gravitational Wave")
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)

        plt2 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit",
                    xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")
        plot!(plt2, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit",
              xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")

        plt3, plt4 = plot_metric_components(p, 13, 40)

        plt5 = plot(losses, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss Curve", legend=false, linewidth=2)

        iron_man_dashboard = plot(plt1, plt2, plt3, plt5, layout = (2, 2), 
                                    size = (1000, 800), 
                                    plot_title = "\nOptimization Iteration: $(length(losses)) | Loss: $(round(loss_val, digits = 5))",
                                    plot_titlevspan = 0.1, 
                                    margin = 8mm)
        display(iron_man_dashboard)

        return false
    end

    # global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 10
        # FIX: Changed stepnorm from 10 to 0.01
        print("i:", i, ", lr: 0.01 (Corrected)\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 20)
                                 
    elseif i == 10
        # FIX: Changed stepnorm from 5 to 0.01
        print("i:", i, ", lr: 0.01\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false,
                                 maxiters = 20)
                                 
    elseif i == optimization_increments[end-3]  # 97
        # FIX: Changed stepnorm from 2 to 0.005
        print("i:", i, ", lr: 0.005\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false,
                                 maxiters = 20)
                                 
    elseif i == optimization_increments[end-2]  # 98
        # FIX: Changed stepnorm from 1 to 0.001
        print("i:", i, ", lr: 0.001\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=0.1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false,
                                 maxiters = 10)
                                 
    elseif i == optimization_increments[end-1]  # 99
        # FIX: Changed stepnorm from 1 to 0.001
        print("i:", i, ", lr: 0.001\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=0.01, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false,
                                 maxiters = 3)
    else  # 100
        print("i:", i, ", lr: 0.001\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=0.001, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 maxiters = 30, 
                                 allow_f_increases=false, 
                                 iterations = 1)
    end

    global NN_params = res.minimizer

    percent_data = round(frac*100; digits = 1)
    current_loss = length(losses) > 0 ? losses[end] : 0.0

    next!(progress_bar; showvalues = [
        (:Increment, "$(i) / $(optimization_increments[end])"),
        (:Data_Size, "$(percent_data)%"),
        (:Current_Loss, current_loss)
    ])
end

""" ********************************
STEP 11: VISUALIZE METRIC COMPONENTS
******************************** """

NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)

print("********************************\nFinal Params:", NN_params.conservative.layer_3)

u0_opt = make_u0(NN_params)
prob_opt = ODEProblem(ODE_model, u0_opt, tspan, NN_params)
optimized_solution = solve(prob_opt, Tsit5(), saveat = tsteps, dt = dt, adaptive=false)
pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)
pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

plot(pred_orbit2_nn[1,:], label = "Optimized Prediction Real")
plot_metric_components(NN_params, 13, 40)
plot(losses, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss Curve", legend=false, linewidth=2)
plot(pred_waveform_real, label = "Optimized Prediction Real")
plot(waveform_real_ecc, label = "True Waveform Real")