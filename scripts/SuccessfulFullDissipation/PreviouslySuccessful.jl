```
    Nested Inverse Problem: Gravitational Wave to Orbits to Metric
    Learns Only g_{tt} component of Schwarzschild Metric + Orbits + Waveform for Conservative Dynamics!

```

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

include("TrainingDataFunctions.jl")
include("PreviousNNFunctions.jl")
include("Orbit2Waveform.jl")
include("DissipationRicci.jl")

gr()

""" ************************
STEP 1: IMPORT TRAINING DATA
************************ """
tspan = (0, 2e3)
datasize = 500
tsteps = range(tspan[1], tspan[2], length = datasize) 

# Gather waveform data
x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p10_e0.4.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p10_e0.4.txt")
waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p10_e0.4.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p10_e0.4.txt")
true_r = file2waveform(tsteps,"input/r_variable_Schwarzschild_p10_e0.4.txt")
true_ϕ = file2waveform(tsteps,"input/phi_variable_Schwarzschild_p10_e0.4.txt")

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
p = 10
semilatus_rectum = p 
e = 0.4

r_min = p / (1+e)
r_max = p / (1-e)

E0, L0 = pe_2_EL(p, e)[2:3] # Schwarzschild IC
r0 = p / (1 - e)

""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """

NN_Conservative = Chain(
    Dense(1, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 2),
)

NN_Dissipative = Chain(
    Dense(1, 10, tanh), # Input: r
    Dense(10, 10, tanh),
    Dense(10, 1),
)

""" *******************************
STEP 5: INITIALIZE NEURAL NETWORKS
******************************* """
# Initialize parameters for both NNs
rng = MersenneTwister(222)
NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)
NN_Dissipative_params, NN_Dissipative_state = Lux.setup(rng, NN_Dissipative)

initialize_Conservative_NN(1e-2, -3, -1, -1)
initialize_Dissipative_NN(1e-2, -3, -1, -1)

""" ******************************************************************
STEP 6: ASSIGN NEURAL NETWORK INPUTS & EXTRACT NEURAL NETWORK OUTPUTS
****************************************************************** """
NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)


""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """
function make_u0(params)
    return [
        0.0, r0, pi/2, 0.0,
        -E0, 0.0, 0.0, L0, 0.0
    ]
end

""" ********************************
STEP 7: CREATE FUNCTION FOR ODE MODEL
*********************************"""
function ODE_model(du, u, p, t)
    du = Newtonian(du, u, model_params, t,
                    NN=NN_adapter, NN_params=p)
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

orbit_plot, h₊_plot, hₓ_plot = plot_orbits_waveforms()
display(orbit_plot)
display(h₊_plot)
display(hₓ_plot)

g_tt_plot, g_rr_plot = plot_metric_components(NN_params, r_min, r_max)
display(g_tt_plot)
display(g_rr_plot)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """

rm("training_log.txt", force=true)

function loss(NN_params; saveat=tsteps, ricci_weight = 0.0)
    tspan = (saveat[1],saveat[end])

    # WAVEFORM LOSS
    u0_local = make_u0(NN_params)
    prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = solve(prob_pred, Tsit5(); saveat = saveat, dt = dt, adaptive=false, verbose = false)
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    N = length(pred_waveform_real)
    waveform_loss = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N])) +
                    ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))

    # RICCI LOSS
    r_check = range(r_min, r_max, length = 10)
    ricci_loss = 0

    function metric(x)
        r_val = x[2]
        output, _ = NN_Conservative([r_val / scale_factor], NN_params.conservative, NN_Conservative_state)
        return g_NeuralNetwork(x, output)
    end

    for radius in r_check
        adaptive_weight = (20 / radius) ^ 2
        input_state = @SVector [0, radius, π/2, 0.0]
        g_inv = inv(metric(input_state))

        Riemann_Tensor = Riemann(metric, input_state)
        Ricci_covariant = RicciTensor(Riemann_Tensor) # R_{μν}
        Ricci_contravariant = g_inv * Ricci_covariant * g_inv # R^{μν}
        Ricci_Frobenius = sum(Ricci_covariant .* Ricci_contravariant)

        ricci_loss += adaptive_weight * (Ricci_Frobenius)^2
    end

    ricci_loss = 9e8 * ricci_loss / length(r_check)

    val_wave = ForwardDiff.value(waveform_loss)
    val_ricci = ForwardDiff.value(ricci_loss)

    loss = waveform_loss + ricci_loss

    return (loss, val_wave, val_ricci), pred_waveform_real, pred_waveform_imag, pred_soln
end

# Test loss function
loss(NN_params)
losses = []
waveform_loss = []
ricci_losses = []

max_total_iterations = 1200

""" *************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """

optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
progress_bar = Progress(length(optimization_increments);
                        dt = 1.0, color=:green, desc = "Optimization Progress", 
                        barlen = 40)

open("training_log.txt", "w") do io
    write(io, "\n===============================================\n")
    write(io, "Initiating Optimization Process ... \n")
    write(io, "===============================================\n")
end

for i in optimization_increments
    println("\noptimization increment :: ", i, " of ", num_optimization_increments)
    opt_first = optimization_increments[1]         # 40
    opt_last  = optimization_increments[end]       # 100
    frac = 0.10 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]

    open("training_log.txt", "a") do io  # Changed "w" to "a" to append, not overwrite
        println(io, "\n===============================================")
        println(io, "Training increment ", i, "/", num_optimization_increments,
                ": using ", length(tsteps_increment), " of ", length(tsteps),
                " points (", round(frac*100; digits=1), "% of total data)")
        println(io, "===============================================")
    end

    tmp_loss(p) = loss(p, saveat = tsteps_increment, ricci_weight = current_ricci_weight)
    function scalar_loss(p)
        (loss_val, _, _), _, _, _  = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
        (loss_val, v_wave, v_ricci), pred_waveform_real, _, pred_soln  = tmp_loss(p)
    
        push!(losses, loss_val)
        push!(waveform_loss, v_wave + 1e-20)
        push!(ricci_losses, v_ricci + 1e-20)

        if length(losses) >= max_total_iterations
            print("Reached Maximum Total Iterations of ", max_total_iterations, ". Stopping Optimization.")
            return true
        end

        function metric(x)
            r_val = x[2]
            scale_factor = r_max
            output, _ = NN_Conservative([r_val / scale_factor], p.conservative, NN_Conservative_state)
            return g_NeuralNetwork(x, output)
        end
        
        sample_vector = @SVector [0, 10.0, π/2, 0.0]
        b_track = ForwardDiff.value.(p.conservative.layer_3.bias)
        pred_metric = metric(sample_vector)
        true_metric = g_Schwarzschild(sample_vector)

        open("training_log.txt", "a") do io
            println(io, "----------------Iteration: $(length(losses))-----------------")
            println(io, "Loss: $(loss_val)")
            println(io, "Waveform Loss: $(v_wave)")
            println(io, "Ricci Loss: $(v_ricci)")
            println(io, "Conservative NN Layer 3 Biases: $(b_track)")
            println(io, "Predicted Metric at r = 10: $(diag(pred_metric))")
            println(io, "g_tt at r = 10, 20, 30, 40: $(diag(metric(@SVector [0.0,10.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,20.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,30.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,40.0,π/2,0.0]))[1])")
            println(io, "g_rr at r = 10, 20, 30, 40: $(diag(metric(@SVector [0.0,10.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,20.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,30.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,40.0,π/2,0.0]))[2])")
            println(io, "True Metric at r = 10: $(diag(true_metric))")
            println(io, "NN g_tt correction at r = 10: $(pred_metric[1,1] / true_metric[1,1])")
            println(io, "NN g_rr correction at r = 10: $(pred_metric[2,2] / true_metric[2,2])")
            println(io, "-----------------------------------------------\n")
        end

        N = length(pred_waveform_real)
        startPoint = 1

        # ------ PREDICTED GRAVITATIONAL WAVE -----
        plt1 = plot_waveform(tsteps, waveform_real_ecc, pred_waveform_real, startPoint, N)
        
        # ------ PREDICTED ORBITS -----
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)

        plt2 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit",
                    xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")
        plot!(plt2, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit",
              xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")

        # ------ PREDICTED r ------
        r = pred_soln[2, :]
        plt_r = plot(tsteps[startPoint:N], r[startPoint:N], lw=2, alpha=1, label="Predicted",
                     xaxis = L"t", ylabel = L"r(t)", title = "Radial State Variable")
        plot!(plt_r, tsteps[startPoint:N], true_r[startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="True",
              xaxis = L"t", ylabel = L"r(t)")
        
        # ----- PREDICTED ϕ -----
        ϕ = pred_soln[4, :]
        plt_ϕ = plot(tsteps[startPoint:N], ϕ[startPoint:N], lw=2, alpha=1, label="Predicted",
                     xaxis = L"t", ylabel = L"ϕ(t)", title = "Angular State Variable")
        plot!(plt_ϕ, tsteps[startPoint:N], true_ϕ[startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="True",
              xaxis = L"t", ylabel = L"ϕ(t)")

        # ------ PREDICTED METRICS -----
        plt3, plt4, plt5 = plot_metric_components(p, r_min, r_max)

        # ------ LOSS CURVE -----
        plt6 = plot(losses, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss Curve", legend=true, linewidth=2, primary=false, ylims = (1e-7, 1e1))
        plot!(plt6, waveform_loss, label = "Waveform Loss", linewidth=2)
        plot!(plt6, ricci_losses, label = "Ricci Loss", linewidth = 2)

        # ------ 3D ORBITS -----
        plt7 = plot3d(pred_orbit1[1, startPoint:N], pred_orbit1[2, startPoint:N], tsteps[startPoint:N], label = "Pred BH 1", linewidth = 2, title = "3D Orbits")
        plot3d!(pred_orbit2[1, startPoint:N], pred_orbit2[2, startPoint:N], tsteps[startPoint:N], label = "Pred BH 2", linewidth = 2)
        plot3d!(x_ecc[startPoint:N], y_ecc[startPoint:N], tsteps[startPoint:N], ls=:dash, label = "True BH 1", linewidth = 2)
        plot3d!(x2_ecc[startPoint:N], y2_ecc[startPoint:N], tsteps[startPoint:N], ls=:dash, label = "True BH 2", linewidth = 2)

        # ------ ALL PLOTS ------
        iron_man_dashboard = plot(plt1, plt2, plt7, plt3, plt4, plt5, plt6, plt_r, plt_ϕ, layout = (3, 3), 
                                    size = (1300, 1000), 
                                    plot_title = "\nOptimization Iteration: $(length(losses)) | Loss: $(round(loss_val, digits = 5))",
                                    plot_titlevspan = 0.1, 
                                    margin = 2mm, 
                                    margin_left = 3mm)
        display(iron_man_dashboard)

        return false
    end

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i == 1
        current_ricci_weight = 0
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        res = Optimization.solve(optprob,
                                 Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 70)
    elseif i == 2
        current_ricci_weight = 1e-4
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 80)
    elseif i == 3
        current_ricci_weight = 1e-3
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 130)
    elseif i == 4
        current_ricci_weight = 1e-2
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=0.5, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 120)
    elseif i == 5
        current_ricci_weight = 1e-1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight)
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 120)
    elseif i == 6
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight)
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 120)
    elseif i == 7
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight)
        res = Optimization.solve(optprob, 
                                Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), 
                                callback=opt_callback, 
                                allow_f_increases=false, maxiters = 120)
    else
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight)
        res = Optimization.solve(optprob, 
                                Optim.BFGS(; initial_stepnorm=1e-4, linesearch = LineSearches.BackTracking()), 
                                callback=opt_callback, 
                                allow_f_increases=false, maxiters = 120)
    end

    global NN_params = res.minimizer

    if length(losses) >= max_total_iterations
        break
    end

    percent_data = round(frac*100; digits = 1)
    current_loss = length(losses) > 0 ? losses[end] : 0.0

    next!(progress_bar; showvalues = [
        (:Increment, "$(i) / $(optimization_increments[end])"),
        (:Data_Size, "$(percent_data)%"),
        (:Current_Loss, current_loss)
    ])
end

w1, b1, w2, b2, w3, b3 = get_weights_biases(NN_params)