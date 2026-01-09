```

    Inverse problem script to learn ODE model from SXS waveform data

```

include("PreviousNNFunctions.jl")
include("Orbit2Waveform.jl")
include("DissipationRicci.jl")

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
x_ecc, y_ecc = file2trajectory(tsteps,"input/dissipative_trajectoryA_Schwarzschild_p20_e0.txt")
x2_ecc, y2_ecc = file2trajectory(tsteps,"input/dissipative_trajectoryB_Schwarzschild_p20_e0.txt")
waveform_real_ecc = file2waveform(tsteps,"input/dissipative_waveform_real_Schwarzschild_p20_e0.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/dissipative_waveform_imag_Schwarzschild_p20_e0.txt")

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
semilatus_rectum = p # semilatus rectum
e = 0.0

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

for (i, layer) in enumerate(NN_Dissipative_params)
    if ~isempty(layer)
        if i == length(NN_Dissipative_params)  # Final layer
            layer.weight .= 1e-2 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= -3
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
    scale_factor = semilatus_rectum

    # Conservative network
    conservative_features = [u[2] / scale_factor]
    conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)
    
    # Dissipative network
    dissipative_features = [u[2] / scale_factor]  # t, r, θ, φ, p_t, p_r, p_θ, p_ϕ
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

rm("training_log.txt", force=true)

function loss(NN_params; saveat=tsteps)
    tspan = (saveat[1],saveat[end])

    u0_local = make_u0(NN_params)
    prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = solve(prob_pred, Tsit5(); saveat = saveat, dt = dt, adaptive = false, verbose = false)
    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    N = length(pred_waveform_real)

    waveform_loss = ( sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real)) / (sum(abs2, waveform_real_ecc[1:N])) +
                    ( sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag)) / (sum(abs2, waveform_imag_ecc[1:N]))
    
    # tikhonov_reg = 1e-10 * sum(abs2, NN_params)
    tikhonov_reg_conservative = 1e-10 * sum(abs2, NN_params.conservative)
    tikhonov_reg_dissipative = 1e-10 * sum(abs2, NN_params.dissipative)  # 1000x stronger!
    tikhonov_reg = tikhonov_reg_conservative + tikhonov_reg_dissipative

    r_check = range(13, 40, length = 10)
    
    ricci_frobenius_loss = 0.0
    ricci_frobenius_weighted = 0.0

    scale_factor = semilatus_rectum

    if pred_soln.retcode != SciMLBase.ReturnCode.Success || any(!isfinite, Array(pred_soln))
        big = 1f6
        return (big, big, 0f0, 0f0), pred_waveform_real*0, pred_waveform_imag*0, pred_soln
    end

    if any(!isfinite, pred_waveform_real) || any(!isfinite, pred_waveform_imag)
        big = 1f6
        return (big, big, 0f0, 0f0), pred_waveform_real*0, pred_waveform_imag*0, pred_soln
    end

    function metric(x)
        r_val = x[2]
        output, _ = NN_Conservative([r_val / scale_factor], NN_params.conservative, NN_Conservative_state)
        return g_NeuralNetwork(x, output)
    end

    for r_val in r_check

        input_val = @SVector [0.0, r_val, π/2, 0.0]

        Riemann_Tensor = Riemann(metric, input_val)
        Ricci_cov = RicciTensor(Riemann_Tensor) # R_{μν}
        g_inv = inv(metric(input_val))

        Ricci_contravariant = g_inv * Ricci_cov * g_inv # R^{μν}

        adaptive_weight = (20.0 / r_val)^2 

        ricci_frobenius_sq = sum(Ricci_cov .* Ricci_contravariant)
        ricci_frobenius_loss += ricci_frobenius_sq^2
        ricci_frobenius_weighted += adaptive_weight * ricci_frobenius_sq^2
    end

    ricci_frobenius_loss /= length(r_check)
    ricci_frobenius_weighted /= length(r_check)

    ricci_frobenius_loss = 4e9 * ricci_frobenius_loss
    ricci_frobenius_weighted *= 9e8

    val_wave = ForwardDiff.value(waveform_loss)
    val_tikh = ForwardDiff.value(tikhonov_reg)
    val_ricci_frobenius = ForwardDiff.value(ricci_frobenius_weighted)

    loss = waveform_loss + tikhonov_reg + ricci_frobenius_weighted

    return (loss, val_wave, val_tikh, val_ricci_frobenius), pred_waveform_real, pred_waveform_imag, pred_soln
end

# Add this function after your loss function
# function debug_dissipation_coefficients(NN_params, r_values=[40.0, 30.0, 20.0, 15.0, 10.0, 5.0])
#     println("\n========== DISSIPATION COEFFICIENTS DEBUG ==========")
    
#     # Get coefficients at different radii
#     for r_test in r_values
#         u_test = [0.0, r_test, π/2, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]
#         nn_out = NN_adapter(u_test, NN_params)
        
#         dissipation_raw = nn_out.dissipative
#         C1 = softplus(dissipation_raw[1])
#         C2 = softplus(dissipation_raw[2])
#         C3 = softplus(dissipation_raw[3])
#         C4 = softplus(dissipation_raw[4])
        
#         # Compute total flux at this radius
#         total_flux = C1 + C2/r_test + C3/r_test^2 + C4/r_test^3
        
#         # For comparison with true value (0.0103/r for your synthetic data)
#         true_flux = 0.0103 / r_test
        
#         println("  At r=$(r_test):")
#         println("    C1 = $(round(C1, sigdigits=4))")
#         println("    C2 = $(round(C2, sigdigits=4))")
#         println("    C3 = $(round(C3, sigdigits=4))")
#         println("    C4 = $(round(C4, sigdigits=4))")
#         println("    Total flux = $(round(total_flux, sigdigits=4)) (true: $(round(true_flux, sigdigits=4)))")
#         println("    Error = $(round(abs(total_flux - true_flux), sigdigits=3))")
#         println()
#     end
    
#     # Print the coefficients as they would appear in the equation
#     u_sample = [0.0, 20.0, π/2, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]
#     nn_out = NN_adapter(u_sample, NN_params)
#     C1 = softplus(nn_out.dissipative[1])
#     C2 = softplus(nn_out.dissipative[2])
#     C3 = softplus(nn_out.dissipative[3])
#     C4 = softplus(nn_out.dissipative[4])
    
#     println("  Dissipation formula:")
#     println("    F(r) = $(round(C1, sigdigits=4)) + $(round(C2, sigdigits=4))/r + $(round(C3, sigdigits=4))/r² + $(round(C4, sigdigits=4))/r³")
#     println("====================================================\n")
# end

# Test loss function
loss(NN_params)

losses = []
waveform_loss = []
tikhonov_loss = []
ricci_frobenius_loss = []

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

    progress_fraction = (i - opt_first) / (opt_last - opt_first)
    # frac = 0.10 + 0.90 * progress_fraction^3  # Cubic: even more end-heavy
    frac = 0.10 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]
    println("Training increment ", i, "/", num_optimization_increments,
        ": using ", length(tsteps_increment), " of ", length(tsteps),
        " points (", round(frac*100; digits=1), "% of total data)")
    open("training_log.txt", "a") do io
        println(io, "================ Optimization Increment: $(i) / $(num_optimization_increments) ================")
        println(io, "Using $(length(tsteps_increment)) of $(length(tsteps)) data points (", round(frac*100; digits=1), "% of total data)")
    end

    tmp_loss(p) = loss(p,saveat=tsteps_increment)
    
    function scalar_loss(p)
        (loss_val, _, _, _), _, _, _  = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function opt_callback(state, args...; kwargs...)
        p = state.u
        # print("\n<------- NN_Conservative layer 3 weights and biases ------->:", p.conservative.layer_3)
        (loss_val, v_wave, v_tikh, v_ricci_frobenius), pred_waveform_real, pred_waveform_imag, pred_soln  = tmp_loss(p)
    
        b_track = ForwardDiff.value.(p.conservative.layer_3.bias)
        dissipation_bias_track = ForwardDiff.value.(p.dissipative.layer_3.bias)
        sample_vector = @SVector [0, 10.0, π/2, 0.0]

        push!(losses, loss_val)
        push!(waveform_loss, v_wave + 1e-20)
        push!(tikhonov_loss, v_tikh + 1e-20)
        push!(ricci_frobenius_loss, v_ricci_frobenius + 1e-20)

        print("\nIteration $(length(losses)) | Loss: $(round(loss_val, digits = 5))")
    
        function metric(x)
            r_val = x[2]
            scale_factor = semilatus_rectum
            output, _ = NN_Conservative([r_val / scale_factor], p.conservative, NN_Conservative_state)
            return g_NeuralNetwork(x, output)
        end

        pred_metric = metric(sample_vector)
        true_metric = g_Schwarzschild(sample_vector)
    
        # if length(losses) % 10 == 0  # Every 10 iterations
        #     debug_dissipation_coefficients(p)
        # end

        open("training_log.txt", "a") do io
            println(io, "----------------Iteration: $(length(losses))-----------------")
            println(io, "Loss: $(loss_val)")
            println(io, "Waveform Loss: $(v_wave)")
            println(io, "Tikhonov Regularization: $(v_tikh)")
            println(io, "Ricci Tensor Frobenius Loss: $(v_ricci_frobenius)")
            println(io, "Conservative NN Layer 3 Biases: $(b_track)")
            println(io, "Dissipative NN Layer 3 Biases: $(dissipation_bias_track)")
            println(io, "Predicted Metric at r = 10: $(diag(pred_metric))")
            println(io, "g_tt at r = 10, 20, 30, 40: $(diag(metric(@SVector [0.0,10.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,20.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,30.0,π/2,0.0]))[1]), $(diag(metric(@SVector [0.0,40.0,π/2,0.0]))[1])")
            println(io, "g_rr at r = 10, 20, 30, 40: $(diag(metric(@SVector [0.0,10.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,20.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,30.0,π/2,0.0]))[2]), $(diag(metric(@SVector [0.0,40.0,π/2,0.0]))[2])")
            println(io, "g_ϕϕ at r = 10, 20, 30, 40: $(diag(metric(@SVector [0.0,10.0,π/2,0.0]))[4]), $(diag(metric(@SVector [0.0,20.0,π/2,0.0]))[4]), $(diag(metric(@SVector [0.0,30.0,π/2,0.0]))[4]), $(diag(metric(@SVector [0.0,40.0,π/2,0.0]))[4])")
            println(io, "True Metric at r = 10: $(diag(true_metric))")
            println(io, "NN g_tt correction at r = 10: $(pred_metric[1,1] / true_metric[1,1])")
            println(io, "NN g_rr correction at r = 10: $(pred_metric[2,2] / true_metric[2,2])")
            println(io, "NN g_ϕϕ correction at r = 10: $(pred_metric[4,4] / true_metric[4,4])")
            for r_test in [6.0, 10.0, 20.0, 40.0]
                u_sample = [0.0, r_test, π/2, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]
                nn_out = NN_adapter(u_sample, p)
                flux_nn = 1e-2 * softplus(nn_out.dissipative[1])
                flux_true = 0.0103 / r_test
                error_pct = 100 * abs(flux_nn - flux_true) / flux_true
                println(io, "r=$(r_test): NN flux=$(round(flux_nn, sigdigits=4)), True=$(round(flux_true, sigdigits=4)), Error=$(round(error_pct, digits=1))%")
            end
        end

        

        N = length(pred_waveform_real)
        startPoint = 1
        plt1 = plot(tsteps[startPoint:N], waveform_real_ecc[startPoint:N], 
                    markershape=:circle, markeralpha = 0.25,
                    linewidth = 2, alpha = 0.5, label = "True",
                    xaxis = "Time", ylabel = L"h(t)", title = "Gravitational Wave")
        plot!(plt1, tsteps[startPoint:N], pred_waveform_real[startPoint:N],
              markershape=:circle, markeralpha = 0.25,
              linewidth = 2, alpha = 0.5, label = "Predicted",
              xaxis = "Time", ylabel = L"h(t)", title = "Gravitational Wave")
        pred_orbit_relative = soln2orbit(pred_soln)
        pred_orbit1, pred_orbit2 = one2two(pred_orbit_relative, 1, mass_ratio)

        plt2 = plot(x_ecc[startPoint:N], y_ecc[startPoint:N], lw=2, alpha=1, label="True Orbit",
                    xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")
        plot!(plt2, pred_orbit1[1,startPoint:N], pred_orbit1[2,startPoint:N], lw=2, alpha=0.9, ls=:dash, color=:red, label="Predicted Orbit",
              xaxis = L"x", ylabel = L"y", title = "Orbital Trajectory")

        plt3, plt4 = plot_metric_components(p, 13, 40)

        plt5 = plot(losses, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss Curve", linewidth=2, label = "Total Loss")
        plot!(plt5, waveform_loss, label="Waveform Loss", linewidth=2)
        plot!(plt5, tikhonov_loss, label="Tikhonov Loss", linewidth=2)
        plot!(plt5, ricci_frobenius_loss, label="Ricci Frobenius Loss", linewidth=2)

        plt6 = plot(ricci_frobenius_loss, yscale=:log10, xlabel="Iteration", ylabel="Ricci Frobenius Loss", title=L"R^{\mu\nu}R_{\mu\nu}", legend=false, linewidth=2)

        orbit_3d = plot3d(
            pred_orbit1[1,:], pred_orbit1[2,:], tsteps_increment,
            linewidth = 2,
            label = "Predicted",
            legend = :topleft,
            xaxis = L"x",
            ylabel = L"y",
            zaxis = "Time",
            title = "3D Orbital Trajectory", 
            ls =:dash
        )

        plot3d!(x_ecc[1:N], y_ecc[1:N], tsteps_increment,
            linewidth = 2,
            label = "True",
        )

        plot3d!(orbit_3d, 
            x2_ecc[1:N], y2_ecc[1:N], tsteps_increment,
            linewidth = 2,
            label = "True BH 2",
        )

        plot3d!(orbit_3d, 
            pred_orbit2[1,:], pred_orbit2[2,:], tsteps_increment,
            linewidth = 2,
            label = "Predicted BH 2",
            ls =:dash
        )

        iron_man_dashboard = plot(plt1, plt2, plt3, plt5, plt6, orbit_3d, layout = (3, 2), 
                                    size = (1000, 1000), 
                                    plot_title = "\nOptimization Iteration: $(length(losses)) | Loss: $(round(loss_val, digits = 5))",
                                    plot_titlevspan = 0.1, 
                                    margin = 2mm)
        display(iron_man_dashboard)

        return false
    end

    # global NN_params = Lux.fmap(x -> x .+ Float64(1e-5)*randn(eltype(x), size(x)), NN_params) 

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i == 1
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=5, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 60)
    elseif i == 2
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 60)
    elseif i == 3
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 60)
    elseif i == 4
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 30)
    elseif i == 5
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 20)
    elseif i == 6
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-1, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 40)
    elseif i == 7
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-2, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 60)
    elseif i == 8
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 80)
    elseif i == 9
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 80)
    else
        res = Optimization.solve(optprob, 
                                 Optim.BFGS(; initial_stepnorm=1e-3, linesearch = LineSearches.BackTracking()), 
                                 callback=opt_callback, 
                                 allow_f_increases=false, maxiters = 60)
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