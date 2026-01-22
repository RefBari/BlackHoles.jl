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
using Optimization
using OptimizationOptimisers
using Plots
using Lux
using Zygote


gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ModelsGeneralized.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

""" *******************************
STEP 1: DEFINE SIMULATION PARAMETERS
******************************* """
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (0, 9300)
datasize = 500
tsteps = range(tspan[1], tspan[2], length = datasize) 
dt_data = tsteps[2] - tsteps[1]
dt = 1.0
num_optimization_increments = 20

""" ************************
STEP 2: IMPORT TRAINING DATA
************************ """
# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p40_e0p2.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p40_e0p2.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p40_e0p2.txt")

""" *******************************
STEP 3: DEFINE INITIAL CONDITIONS
******************************* """
p = 40
e = 0.2

# E0, L0 = circular_pt_L(R)
# E0, L0 = eccentric_pt_L(p, e) # Newtonian IC
E0, L0 = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0 = p / (1 - e)
pᵣ0 = 0.0
u0 = [0.0, r0, pi/2, 0.0, E0, pᵣ0, 0.0, L0, 0.0]


""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """
#  # Neural network setup 
NN_Conservative = Chain(
    Dense(1, 30, tanh),
    Dense(30, 30, tanh),
    Dense(30, 1),
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
    dissipative = NN_Dissipative_params
)

"""
STEP 7: CREATE FUNCTION FOR ODE MODEL
"""
function ODE_model_dual(du, u, p, t)
    du = GENERIC(du, u, model_params, t,
                                    NN=NN_adapter_dual, 
                                    NN_params=p)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
************************************************************"""
prob_nn_dual = ODEProblem(ODE_model_dual, u0, tspan, NN_params)
soln_nn_dual = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
waveform_nn_real_dual, waveform_nn_imag_dual = compute_waveform(dt_data, soln_nn_dual, mass_ratio; coorbital=false)
orbit = soln2orbit(soln_nn_dual)
pred_orbit1_init, pred_orbit2_init = one2two(orbit, 1, mass_ratio)

plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
plot(waveform_real_ecc, label = "Real")
plot!(waveform_nn_real_dual, label = "Prediction")

""" ********************************************************
STEP 8a: TEST INITIAL SOLUTION BEFORE ANY TRAINING
************************************************************"""
# Define an r range (just outside the Schwarzschild radius)
rvals_init = range(10, 80.0; length=400)

# Evaluate NN-predicted multiplicative corrections
f_tt_pred_init = []
f_rr_pred_init = []
f_pp_pred_init = []

for r in rvals_init
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    # push!(f_tt_pred_init, exp(out[1]))
    # push!(f_rr_pred_init, exp(out[1]))
    push!(f_pp_pred_init, out[1])
end

# f = (1 - (2/r))

# SANITY CHECK: Compute predicted *inverse* Schwarzschild metric components starting w/ Schwarzschild
# g_tt_pred_init = [-(1 - (2/r))^(-1) for (r, f_rr) in zip(rvals_init, f_rr_pred_init)]
# g_rr_pred_init = [(1 - (2/r))*f_rr for (r, f_rr) in zip(rvals_init, f_rr_pred_init)]
# g_ϕϕ_pred_init = [r^(-2)*f_pp for (r, f_pp) in zip(rvals_init, f_pp_pred_init)]

# Compute predicted Schwarzschild metric components starting with Newtonian
# g_tt_pred_init = [-(1 - 2/r)^(-1) for (r, f_rr) in zip(rvals_init, f_rr_pred_init)]
# g_rr_pred_init = [((1 + 2/r)^(-1))*f_rr for (r, f_rr) in zip(rvals_init, f_rr_pred_init)]
g_ϕϕ_pred_init = [r^(-2) * (1 + 2/r)^(-1) * (1 + f_pp) for (r, f_pp) in zip(rvals_init, f_pp_pred_init)]

# Compute true Schwarzschild metric components
f = 1 .- 2 ./ rvals_init
# g_tt_true = -1 ./ f
# g_rr_true = f
g_ϕϕ_true = 1 ./ rvals_init.^2

# plt1 = plot(rvals_init, g_tt_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
# plot!(plt1, rvals_init, g_tt_pred_init, lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{tt}", title=L"g^{tt}(r)")

plt2 = plot(rvals_init, g_rr_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt2, rvals_init, g_rr_pred_init, lw=2, color=:red, label="Initial Prediction", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")

plt3 = plot(rvals_init, g_ϕϕ_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt3, rvals_init, g_ϕϕ_pred_init, lw=2, color=:red, label="Initial Prediction", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")

sol_init = solve(remake(prob_nn_dual, p = NN_params, tspan=tspan), Tsit5(), saveat = tsteps, dt = dt, adaptive=false)

pred_waveform_real_init, pred_waveform_imag_init = compute_waveform(dt_data, sol_init, mass_ratio)

pred_orbit_init = soln2orbit(sol_init)
pred_orbit1_nn_init, pred_orbit2_nn_init = one2two(pred_orbit_init, 1, 1)

plt4 = plot(x_ecc, y_ecc, linewidth = 2, label = "truth")
plot!(plt4, pred_orbit1_nn_init[1,:], pred_orbit1_nn_init[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")
plot(x2_ecc, y2_ecc, linewidth = 2, label = "truth 2")
plot!(pred_orbit2_nn_init[1,:], pred_orbit2_nn_init[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")

layout = @layout [a b; c d]
combined_plot = plot(plt1, plt2, plt3, plt4,
                     layout = layout,
                     size = (900, 700),
                     margin = 5mm)
display(combined_plot)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """
# # ## Define the objective function
function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])
    pred_soln = solve(remake(prob_nn_dual, p = NN_params, tspan=tspan), Tsit5(),
                            saveat = saveat, dt = dt, adaptive=false, verbose = false, sensealg=BacksolveAdjoint(checkpointing=true))
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)

    N = length(pred_waveform_real)
    data_loss = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / N
    reg_loss = 1e-8 * sum(abs2, NN_params)
    print("Data_loss: ", data_loss, " Reg_loss: ", reg_loss, "\n")

    loss = data_loss

    return loss, pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(NN_params)
losses = []

""" *************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """
# # ## Train with BFGS (gives best results because the Newtonian model seems to give a very good initial guess)
optimization_increments = [1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20]
for i in optimization_increments
    println("optimization increment :: ", i, " of ", num_optimization_increments)
    # tsteps_increment = tsteps[tsteps .<= tspan[1]+i*(tspan[2]-tspan[1])/num_optimization_increments]
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
    
        loss_val, pred_waveform_real, pred_waveform_imag, pred_soln = tmp_loss(p)
    
        push!(losses, loss_val)
        display(loss_val)
    
        N = length(pred_waveform_real)
        startPoint = 1
        z_axis  = tsteps[startPoint:N]
        # psuh!(predicted_wave, pred_waveform_real[N])
        # print("real waveform: ", waveform_real_ecc, 
        #     "\npred waveform: ", pred_waveform_real)
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True Gravitational Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted Gravitational Wave")
            
        display(plt1)
        # N = 10
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)
        plt2 = plot3d(x_ecc[startPoint:N], y_ecc[startPoint:N], z_axis;
              lw=2, alpha=1, label="orbit data",
              camera=(35,20))  # azimuth,elevation
        # plot!(plt2, x2_ecc[startPoint:N], y2_ecc[startPoint:N], z_axis;
        #       lw=2, alpha=0.9, color=:orange, label="orbit 2 data")
        plot!(plt2, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], z_axis;
      lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN")
        # plot!(plt2, pred_orbit2[1,startPoint:N], pred_orbit2[2,startPoint:N], z_axis;
        #       linewidth = 2, alpha = 0.8, color = "green",
        #       label = "orbit 2 NN", ls=:dash, aspect_ratio=:equal)
        display(plt2)

        plt3 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N],
        lw=2, alpha=1, label="orbit data",
        camera=(35,20))
        # plot!(plt3, x2_ecc[startPoint:N], y2_ecc[startPoint:N],
        #       lw=2, alpha=0.9, color=:orange, label="orbit 2 data")
        plot!(plt3, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N],
      lw=2, alpha=0.9, ls=:dash, color=:red, label="orbit NN")
        # plot!(plt3, pred_orbit2[1,startPoint:N], pred_orbit2[2,startPoint:N],
        #       linewidth = 2, alpha = 0.8, color = "green",
        #       label = "orbit 2 NN", ls=:dash, aspect_ratio=:equal)
        display(plt3)
        
        r_pred = pred_soln[2,:]
        ϕ_pred = pred_soln[4,:]
        p_r_pred = pred_soln[6,:]
        p_ϕ_pred = pred_soln[8,:]

        plt4 = plot(r_pred, xlabel = L"t", ylabel = L"r", label = "Predicted", legend =:topright, linewidth = 2)
        plt5 = plot(ϕ_pred, xlabel = L"t", ylabel = L"ϕ", label = "Predicted", legend =:topright, linewidth = 2)
        plt6 = plot(p_r_pred, xlabel = L"t", ylabel = L"p_r", label = "Predicted", legend =:topright, linewidth = 2)
        plt7 = plot(p_ϕ_pred, xlabel = L"t", ylabel = L"p_ϕ", label = "Predicted", legend =:topright, linewidth = 2)

        # display(plt4)
        # display(plt5)
        # display(plt6)
        # display(plt7)

        layout = @layout [a b; c d]
        combined_plot = plot(plt4, plt5, plt6, plt7, layout=layout, size=(900, 700), margin=5mm)
        display(combined_plot)

        # layout = @layout [
        #     a; b
        #     ]
    
        # full_plot = plot(plt1, plt2, layout=layout, size = (600, 400))
        # display(plot(full_plot))
        
        return false
    end

    global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i < 10
        print("i:", i, ", lr: 5\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=3, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    elseif i == 10
        print("i:", i, ", lr: 2\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    elseif i == optimization_increments[end-3]  # 97
        print("i:", i, ", lr: 1e-1\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    elseif i == optimization_increments[end-2]  # 98
        print("i:", i, ", lr: 1e-2\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    elseif i == optimization_increments[end-1]  # 99
        print("i:", i, ", lr: 1e-3\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    else  # 100'
        print("i:", i, ", lr: 1e-4\n")
        local res = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), g_tol = 1e-8, callback=opt_callback, maxiters = 30, allow_f_increases=true)
    end

    global NN_params = res.minimizer
    print("The final optimized parameters are:", NN_params)
    # local plt = plot(losses, yaxis=:log, linewidth = 2, = "Iteration", ylabel = "Objective value", legend = false)
    # display(plot(plt))
end

""" ********************************
STEP 11: VISUALIZE METRIC COMPONENTS
******************************** """

# Define an r range (just outside the Schwarzschild radius)
rvals = range(33, 50; length=400)

# Evaluate NN-predicted multiplicative corrections
# f_tt_pred = []
f_rr_pred = []
f_pp_pred = []

for r in rvals
    out, _ = NN_Conservative([r], NN_params.conservative, NN_Conservative_state)
    # push!(f_tt_pred, exp(out[1]))
    # push!(f_rr_pred, exp(out[1]))
    push!(f_pp_pred, out[1])
end

f = 1 .- 2 ./ rvals

# Compute predicted *inverse* Schwarzschild metric components
# g_tt_pred = [-(1 - (2/r))^(-1) for (r, f_rr) in zip(rvals_init, f_rr_pred_init)]
# g_rr_pred = [(1 - (2/r))+f_rr for (r, f_rr) in zip(rvals, f_rr_pred)]

# Schwarzschild
g_rr_true = f
g_ϕϕ_true = 1 ./ rvals.^2

# Predicted 
# g_rr_pred = [((1 + 2/r)^(-1)) * f_rr for (r, f_rr) in zip(rvals, f_rr_pred)]
g_ϕϕ_pred = [r^(-2) * (1 + 2/r)^(-1) * (1 + f_pp) for (r, f_pp) in zip(rvals, f_pp_pred)]

# Newtonian
# g_rr_orig = [((1 + 2/r)^(-1)) for (r, f_rr) in zip(rvals, f_rr_pred)]
g_ϕϕ_orig = [r^(-2) * (1 + 2/r)^(-1) for (r, f_pp) in zip(rvals, f_pp_pred)]

# Plot each component
# plt1 = plot(rvals, g_tt_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
# plot!(plt1, rvals, g_tt_pred, lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{tt}", title=L"g^{tt}(r)")

# plt_t_diff = plot(rvals, g_tt_true.-g_tt_pred, lw=2, ls=:dashdot, color=:green, title="What NN should learn for g^tt", xlabel=L"r", ylabel=L"g_{Schwarzschild}^{tt}-g_{Newton}^{tt}", legend =:false, aspect_ratio =:equal)

# display(plt1)

plt2 = plot(rvals, g_rr_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt2, rvals, g_rr_orig, lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")
plot!(plt2, rvals, g_rr_pred, lw=2, label="Predicted", xlabel=L"r", ylabel=L"g^{rr}", title=L"g^{rr}(r)")
# display(plt2)
plt_r_diff = plot(rvals, g_rr_true.-g_rr_pred, lw=2, ls=:dashdot, color=:green, title="What NN should learn for g^rr", xlabel=L"r", ylabel=L"g_{Schwarzschild}^{rr}-g_{Newton}^{rr}", legend =:false)

plt3 = plot(rvals, g_ϕϕ_true, lw=2, ls=:dash, color=:blue, label="Schwarzschild")
plot!(plt3, rvals, g_ϕϕ_orig, lw=2, color=:red, label="Newtonian", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")
plot!(plt3, rvals, g_ϕϕ_pred, lw=2, label="Predicted", xlabel=L"r", ylabel=L"g^{\phi\phi}", title=L"g^{\phi\phi}(r)")

# plt_ϕ_diff = plot(rvals, g_ϕϕ_true.-g_ϕϕ_pred, lw=2, ls=:dashdot, color=:green, title="What NN should learn for g^ϕϕ", xlabel=L"r", ylabel=L"g_{Schwarzschild}^{ϕϕ}-g_{Newton}^{ϕϕ}", legend =:false)


# display(plt3)

optimized_solution = solve(remake(prob_nn_dual, p = NN_params, tspan=tspan), Tsit5(), saveat = tsteps, dt = dt, adaptive=false)

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, optimized_solution, mass_ratio)

pred_orbit = soln2orbit(optimized_solution)
pred_orbit1_nn, pred_orbit2_nn = one2two(pred_orbit, 1, 1)

plt4 = plot(x_ecc, y_ecc, linewidth = 2, label = "truth")
plot!(plt4, pred_orbit1_nn[1,:], pred_orbit1_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")
plot!(plt4, x2_ecc, y2_ecc, linewidth = 2, label = "truth 2")
plot!(plt4, pred_orbit2_nn[1,:], pred_orbit2_nn[2,:], linestyle = :dash, linewidth = 2, label = "prediction", title = "Orbits")

layout = @layout [a b; c d]
combined_plot = plot(plt1, plt2, plt3, plt4,
                     layout = layout,
                     size = (900, 700),
                     margin = 5mm)
display(combined_plot)

plt5 = plot(losses, yaxis=:log, linewidth = 2, xlabel = "Iteration", ylabel = "Objective value", legend = false)
display(plt5)

"""
STEP 12: SYMBOLIC REGRESSION TO ANALYTICALLY LEARN SCHWARZSCHILD METRIC
"""

# # --- Data (make sure Float64 and finite) ---
# u = collect(1.0 ./ rvals)         # u = 1/r
# g = collect(g_rr_pred)            # your NN-predicted g^{rr}(u)

# mask = isfinite.(u) .& isfinite.(g) .& (abs.(g) .< 1e12)
# u, g = u[mask], g[mask]

# # --- Simple model: ĝ(u; α) = 1 - α u ---
# # Robust squared-error loss
# lossα(θ) = begin
#     α = float(θ[1])
#     pred = 1 .- α .* u
#     r = pred .- g
#     if any(!isfinite, r)
#         return Inf
#     end
#     sum(r.^2)
# end

# # Optional: closed-form LS init to help Adam
# A   = [ones(length(u))  u]
# β   = A \ g
# α0  = -β[2]                         # because g ≈ 1 - α u
# θ0  = [float(α0)]

# # --- Build the optimization problem (use AD for Adam) ---
# optf = Optimization.OptimizationFunction((θ,p)->lossα(θ), Optimization.AutoZygote())
# prob = Optimization.OptimizationProblem(optf, θ0)

# # --- Adam pass ---
# lr = 1e-2
# maxiters = 3000
# res_adam = Optimization.solve(prob, Optimisers.Adam(lr); maxiters=maxiters)

# α_adam = res_adam.u[1]
# @show α_adam res_adam.minimum

# # (Optional) polish with BFGS starting from Adam’s result
# prob2 = Optimization.OptimizationProblem(optf, res_adam.u)
# res_bfgs = Optimization.solve(prob2, Optim.BFGS(); allow_f_increases=true)
# α = res_bfgs.u[1]
# @show α res_bfgs.minimum

# # --- Plot for sanity ---
# using Plots, LaTeXStrings
# g_fit  = 1 .- α .* u
# g_schw = 1 .- 2 .* u
# plt = plot(u, g, lw=2, label="Predicted: g_{NN}^{rr}(u)", color = "red")
# plot!(plt, u, g_fit,  lw=2, ls=:dashdot, label="Parameter Estimation Fit: (1 - αu), α = 1.38", color = "green")
# plot!(plt, u, g_schw, lw=2,  label="Schwarzschild: (1 - 2u)", xlabel = L"u=1/r", ylabel = L"g^{rr}(u)", color = "blue", ls=:dash)
# xlabel(plt, L"u=1/r"); ylabel!(plt, L"g^{rr}(u)"); legend!(plt, :topleft)
# display(plt)
