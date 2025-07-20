using LinearAlgebra
using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures
using Lux
using ComponentArrays
using Random
using ForwardDiff

include(
    "/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/SchwarzschildTrainingData",
)

losses = []
partition_boundaries = []
final_paramaters = []
solutions_list = []
parameters_list = []
final_predicted_h_plus = []
training_h_plus_wave = []

function optimizeBlackHole(;
    learningRate,
    epochsPerIteration,
    numberOfCycles,
    totalTrainingPercent,
    true_parameters,
    initial_guess,
)

    global losses = []
    global partition_boundaries = []
    global final_paramaters = []
    global solutions_list = []
    global parameters_list = []
    global final_predicted_h_plus = []
    global training_h_plus_wave = []

    trainingFraction = totalTrainingPercent # What total fraction of the training data will the neural network learn from?
    p_guess = pe_2_EL(initial_guess[1], initial_guess[2]) # Uses the pe_2_EL function to convert initial guess to (M = 1, E, L)

    true_p = true_parameters[1] # True semi-latus rectum
    true_e = true_parameters[2] # True eccentricity

    # This trainingData returns [prob, true_sol, h_plus_training, h_cross_training]
    trainingData = create_Schwarzschild_trainingData([true_p, true_e]) # Generate Training Data (Gravitational Waveforms)

    timestep = 100 # Timestep for ODE Solver and Optimizer

    true_solution = trainingData[2] # True Solution

    function SchwarzschildHamiltonian_GENERIC(du, u, p, t)
        x = u # u[1] = t, u[2] = r, u[3] = θ, u[4] = ϕ  
        NN_params = p.NN
        M, E, L = p.parameters.M, p.parameters.E, p.parameters.L

        function H(state_vec)
            t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

            # if r < 3.0
            #     # Return a "repulsive" Hamiltonian that pushes particle outward
            #     return 1e6 * (3.0 - r)^2  # Huge energy penalty near r=3
            # end

            NN_correction = NN([r], NN_params, NN_state)[1]

            g_tt_correction = 0.01*NN_correction[1]
            g_rr_correction = 0.01*NN_correction[2]
            g_ϕϕ_correction = 0.001*NN_correction[3]

            p = [p_t, p_r, p_θ, p_ϕ]
            g = [
                -(1 - 2/r)^(-1)+g_tt_correction 0 0 0;
                0 (1 + 2/r)^(-1)+g_rr_correction 0 0;
                0 0 0 0;
                0 0 0 (r^(-2))*(1 + 2/r)^(-1)+g_ϕϕ_correction
            ]

            H_schwarzschild = (1/2) * p' * g * p

            return H_schwarzschild # Returns equations of motion in PROPER time
        end

        # Compute gradient using ForwardDiff
        grad_H = ForwardDiff.gradient(H, x)

        # Define symplectic matrix L (8x8)
        J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

        # Hamilton's equations: ẋ = J*∇H
        du_dτ = J * grad_H

        t_val, r_val = x[1], x[2]
        f_val = 1 - 2*M/r_val
        dτ_dt = f_val/E

        du .= du_dτ .* dτ_dt # du / dt Returns equations of motion in COORDINATE time
    end

    # Neural network setup 
    NN = Chain(
        Dense(1, 4, tanh), # Learns correction term in terms of 4 parameters: r, p_t, p_r, p_ϕ
        Dense(4, 4, tanh),
        Dense(4, 3),
    )        # Output 2 corrections, g^tt and g^rr

    rng = MersenneTwister(222)
    NN_params, NN_state = Lux.setup(rng, NN)
    precision = Float64
    NN_params = Lux.fmap(x -> precision.(x), NN_params)

    # Same weight initialization as your original
    for layer in NN_params
        if ~isempty(layer)
            layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
            layer.bias .= 0.02 * randn(rng, eltype(layer.bias), size(layer.bias))
        end
    end

    # What will the neural network learn? The following parameters: M, E, L, NN_param
    R = initial_guess[1]/(1+initial_guess[2])
    M = p_guess[1]
    E = p_guess[2]
    L = p_guess[3]
    BH_SlightPush = 0

    θ = (; NN = NN_params, parameters = (M = M, E = E, L = L))
    θ = ComponentVector{precision}(θ);

    # Initial State Vector for Particle orbiting Schwarzschild BH
    # u0 = [0, R, pi/2, 0, -1*E, 0, 0, L] # u = [t₀, r₀, θ₀, ϕ₀, pₜ₀, pᵣ₀, p_θ₀, p_ϕ₀]
    # Use relativistic constraint H = -1/2
    g_tt_factor = 1 - 2/R
    g_rr_factor = 1 + 2/R
    angular_term = (L^2 / R^2) * (1/g_rr_factor)
    p_t_squared = g_tt_factor * (1 + angular_term)  # For H = -1/2
    p_t_correct = -sqrt(p_t_squared)

    u0 = [0, R, π/2, 0, p_t_correct, 0, 0, L]
    timeLength = 6e4
    tspan = (0.0, timeLength)
    t = 0:timestep:timeLength

    # Define the ODE Problem using the Neural Network
    prob_learn = ODEProblem(SchwarzschildHamiltonian_GENERIC, u0, tspan, θ)

    function evaluateHamiltonian(u)
        M = 1
        r = u[2]
        f = 1-(2*M/r)
        pₜ = u[5]
        pᵣ = u[6]
        p_ϕ = u[8]
        H = (1/2) * (-f^(-1)*pₜ^2 + f*(pᵣ)^2 + (1/r^2)*(p_ϕ)^2)
        return H
    end

    function loss(pn, trainingFraction)
        newprob = remake(prob_learn, p = pn)
        sol = solve(newprob, RK4(), saveat = timestep)

        predicted_waveform_plus = compute_waveform(timestep, sol, 1.0; coorbital = false)[1]
        predicted_waveform_cross =
            compute_waveform(timestep, sol, 1.0; coorbital = false)[2]

        true_r_values = trainingData[2][2, :]
        predicted_r_values = sol[2, :]

        true_ϕ_values = trainingData[2][4, :]
        predicted_ϕ_values = sol[4, :]

        true_p_ϕ = trainingData[2][8, :]
        predicted_p_ϕ = sol[8, :]

        true_ϕ_dot_values = true_p_ϕ ./ (true_r_values .^ 2)
        predicted_ϕ_dot_values = predicted_p_ϕ ./ (predicted_r_values .^ 2)

        h_plus_training = trainingData[3]
        h_cross_training = trainingData[4]

        # Compare only the overlapping portion
        n_train = Int(floor(length(h_plus_training)*trainingFraction))
        n_pred = length(predicted_waveform_plus)
        n_compare = min(n_pred, n_train)

        orbital_loss =
            2*sum(abs2, predicted_r_values[1:n_compare] .- true_r_values[1:n_compare])
        orbital_loss +=
            8*sum(abs2, predicted_ϕ_values[1:n_compare] .- true_ϕ_values[1:n_compare])
        orbital_loss +=
            9*sum(
                abs2,
                predicted_ϕ_dot_values[1:n_compare] .- true_ϕ_dot_values[1:n_compare],
            )

        waveform_loss =
            sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
        waveform_loss += sum(
            abs2,
            predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare],
        )

        loss_value = waveform_loss + 0.002 * orbital_loss
        loss_value /= n_compare

        println(
            "Training with fraction: ",
            trainingFraction,
            ", n_compare: ",
            n_compare,
            ", loss: ",
            loss_value,
        )
        return loss_value
    end

    function callback(pn, loss; dotrain = true)
        if dotrain
            push!(losses, loss);
            @printf("Epoch: %d, Loss: %15.12f \n", length(losses), loss);
            p = plot(
                losses,
                label = "Loss",
                xlabel = "Epochs",
                ylabel = "Loss",
                top_margin = 10mm,
                bottom_margin = 10mm,
                left_margin = 10mm,
                right_margin = 10mm,
            )
            vline!(partition_boundaries, label = "Partition")
            display(p)
        end
        return false
    end

    # define Optimization Problem
    adtype = Optimization.AutoFiniteDiff() # instead of Optimization.AutoZygote(), use finite differences
    optf = Optimization.OptimizationFunction(
        (x, p) -> loss(x, trainingFraction/numberOfCycles),
        adtype,
    )
    θ_init = θ;

    # create Optimization Problem (function + initial guess for parameters)
    optprob = Optimization.OptimizationProblem(optf, θ_init)

    # choose method for solving problem 
    lr = learningRate;

    # solve the problem
    num_iters = 3; # number of iterations per partition (i.e., 2 partitions means one run + 2 additional runs = 3 runs * 25 epochs/run = 75 epochs)
    opt_result = Optimization.solve(
        optprob,
        Optim.BFGS(; initial_stepnorm = lr),
        callback = callback,
        maxiters = num_iters,
    )
    θ_final = opt_result.u

    NN_params_final = θ_final.NN

    newprob = remake(prob_learn, p = θ_final)
    sol = solve(newprob, RK4(), saveat = timestep)
    push!(solutions_list, sol)

    h_plus_pred = compute_waveform(timestep, sol, 1.0; coorbital = false)[1]
    h_cross_pred = compute_waveform(timestep, sol, 1.0; coorbital = false)[2]

    # Handle all cases properly
    h_plus_training = trainingData[3]
    h_cross_training = trainingData[4]

    n_pred = length(h_plus_pred)
    n_train = length(h_plus_training)

    if n_pred == n_train
        # Same length - no padding needed
        h_plus_pred_plot = h_plus_pred
    elseif n_pred < n_train
        # Predicted is shorter - pad with zeros
        h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
    else  # n_pred > n_train
        # Predicted is longer - truncate
        h_plus_pred_plot = h_plus_pred[1:n_train]
    end

    t_plot = (0:(n_train-1)) * timestep

    p = plot(
        t_plot,
        h_plus_training,
        label = L"$h_+$ true",
        linewidth = 2,
        xlabel = "Time (s)",
        ylabel = L"h_+",
        legend = :topright,
        grid = true,
        top_margin = 10mm,
        bottom_margin = 10mm,
        left_margin = 10mm,
        right_margin = 10mm,
    )
    plot!(t_plot, h_plus_pred_plot, label = L"$h_+$ predicted", linewidth = 2)# plot!(t_plot, predicted_waveform_plus_old_padded, label="h+ initial prediction", linewidth=2)
    display(p)

    function partitionTraining(numCycles, totalTrainingFraction)
        global partition_boundaries, losses, final_paramaters, solutions_list, parameters_list

        amountTrain = totalTrainingFraction / numCycles
        p_final_array = [θ_final]

        for i = 2:numCycles
            trainingFraction = i * amountTrain
            push!(partition_boundaries, length(losses))
            push!(final_paramaters, p_final_array[end])
            optf = Optimization.OptimizationFunction(
                (x, p) -> loss(x, trainingFraction),
                adtype,
            )

            θ_current = p_final_array[end]
            optprob = Optimization.OptimizationProblem(optf, θ_current)
            opt_result_2 = Optimization.solve(
                optprob,
                Optim.BFGS(; initial_stepnorm = lr),
                callback = callback,
                maxiters = num_iters,
            )

            θ_final_2 = opt_result_2.u;
            push!(p_final_array, θ_final_2)
            newprob_2 = remake(prob_learn, p = θ_final_2)
            sol_2 = solve(newprob_2, RK4(), saveat = timestep)
            push!(solutions_list, sol_2)
            push!(parameters_list, getParameters(sol_2))

            # Extract predicted and true solution components
            y1 = sol_2[1, :] # coordinate time solutions
            y2 = sol_2[2, :] # r (radial) coordinate solutions  
            y3 = sol_2[3, :] # equatorial angle solutions
            y4 = sol_2[4, :] # planar angle solutions

            y1_true = true_solution[1, :]
            y2_true = true_solution[2, :]
            y3_true = true_solution[3, :]
            y4_true = true_solution[4, :]

            # Create orbital trajectory plot
            particle_x = y2 .* cos.(y4)
            particle_y = y2 .* sin.(y4)

            particle_x_true = y2_true .* cos.(y4_true)
            particle_y_true = y2_true .* sin.(y4_true)

            orbitalParamsPlot = plot(
                t,
                [y1 y2 y3 y4],
                layout = (2, 2),
                title = ["Coordinate Time" "Radius" "Equatorial Angle" "Angular Displacement"],
                xlabel = "Simulation Time",
                ylabel = [L"t" L"r" L"θ" L"ϕ"],
                label = "Predicted",
                ylims = [(0, maximum(t)) (minimum(y2)-1, maximum(y2)+1) (1.35, 1.8) (
                    0,
                    maximum(y4),
                )],
                bottom_margin = 2mm,
                top_margin = 2mm,
                left_margin = 2mm,
                right_margin = 2mm,
                linewidth = 1,
            )

            plot!(t, [y1_true y2_true y3_true y4_true], linestyle = :dash, label = "True")

            display(orbitalParamsPlot)

            p2 = plot(
                particle_x,
                particle_y,
                aspect_ratio = 1,
                linewidth = 2,
                title = "\n\nPredicted Trajectory",
                xlabel = L"x",
                ylabel = L"y",
                label = "Predicted Trajectory",
                bottom_margin = 10mm,
                top_margin = 10mm,
                left_margin = 10mm,
                right_margin = 10mm,
            )

            plot!(
                particle_x_true,
                particle_y_true,
                linewidth = 2,
                label = "True Trajectory",
                color = "lightsalmon1",
                linestyle = :dash,
            )

            display(p2)

            h_plus_pred = compute_waveform(timestep, sol_2, 1.0; coorbital = false)[1]
            h_cross_pred = compute_waveform(timestep, sol_2, 1.0; coorbital = false)[2]

            # Handle all cases properly
            n_pred = length(h_plus_pred)
            n_train = length(h_plus_training)

            if n_pred == n_train
                # Same length - no padding needed
                h_plus_pred_plot = h_plus_pred
            elseif n_pred < n_train
                # Predicted is shorter - pad with zeros
                h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
            else  # ADD THIS: when n_pred > n_train
                h_plus_pred_plot = h_plus_pred[1:n_train]
            end

            t_plot = (0:(n_train-1)) * timestep

            p = plot(
                t_plot,
                h_plus_training,
                color = "lightsalmon",
                label = "h+ true",
                linewidth = 2,
                xlabel = "Time (s)",
                ylabel = L"h_+",
                legend = :topright,
                grid = true,
                bottom_margin = 10mm,
                top_margin = 10mm,
                left_margin = 10mm,
                right_margin = 10mm,
            )
            plot!(t_plot, h_plus_pred_plot, label = "h+ predicted", color = "blue")# plot!(t_plot, predicted_waveform_plus_old_padded, label="h+ initial prediction", linewidth=2)
            trainingWindow = trainingFraction * timeLength
            vline!(
                [trainingWindow],
                color = "red",
                linestyle = :dash,
                label = "Training Window",
            )
            display(p)

            push!(final_predicted_h_plus, h_plus_pred_plot)
            push!(training_h_plus_wave, h_plus_training)
        end
    end

    numCycles = numberOfCycles

    partitionTraining(numCycles, trainingFraction)

    println("\n=== Neural Network Learned Corrections ===")
    test_r_values = 6.0:0.5:10.0
    println("r    | NN_1      | NN_2      | Target_1  | Target_2")
    println("-----|-----------|-----------|-----------|----------")
    for r_test in test_r_values
        ŷ = NN([r_test], solutions_list[end].prob.p.NN, NN_state)[1]  # Use final trained parameters
        target_1 = (2*M/r_test)/(1-2*M/r_test)  # Should be ≈ -2M/r for weak field  
        target_2 = -2*M/r_test                   # Should be -2M/r
        println(
            "$(round(r_test, digits=1)) | $(round(ŷ[1], digits=3))     | $(round(ŷ[2], digits=3))     | $(round(target_1, digits=3))     | $(round(target_2, digits=3))",
        )
    end
    println("===============================================\n")

    x = range(6, 12, length = 20)
    y = (x .- 6) ./ 2
    p = plot(
        x,
        y,
        ylims = (-0.1, 1),
        xlims = (6, 12),
        linewidth = 3,
        bottom_margin = 10mm,
        top_margin = 10mm,
        left_margin = 10mm,
        right_margin = 10mm,
        label = "Separatrix",
        xlabel = "p (Semi-latus Rectum)",
        ylabel = "e (Eccentricity)",
        legend = :bottomright,
    )
    scatter!(
        [getParameters(true_solution)[1]],
        [getParameters(true_solution)[2]],
        color = "lightsalmon",
        markersize = 5,
        label = "True Parameters",
    )

    # for i in 1:numCycles
    #     scatter!([parameters_list[i][1]], [parameters_list[i][2]], color = "darkseagreen1", markersize = 3, legend = false)
    # end

    display(p)

    final_sol = solutions_list[end]

    t_vals = final_sol[1, :]
    pₜ_vals = final_sol[5, :]
    p_ϕ_vals = final_sol[8, :]

    pₜ_plot = plot(
        t_vals,
        pₜ_vals,
        title = "Energy Conservation",
        xlabel = "Time",
        ylabel = L"p_t",
        label = L"p_t",
        linewidth = 2,
        color = "red",
        ylims = [-0.9738, -0.96],
    )
    p_ϕ_plot = plot(
        t_vals,
        p_ϕ_vals,
        title = "Angular Momentum Conservation",
        xlabel = "Time",
        ylabel = L"p_φ",
        label = L"p_φ",
        linewidth = 2,
        color = "blue",
        ylims = [0.9*minimum(p_ϕ_vals), 1.1*maximum(p_ϕ_vals)],
    )

    conservedQuantities = plot(pₜ_plot, p_ϕ_plot, layout = (2, 1))

    display(conservedQuantities)

    return (parameters_list[end][1] - getParameters(true_solution)[1])^2 +
           (parameters_list[end][2] - getParameters(true_solution)[2])^2
end

optimizeBlackHole(
    learningRate = 9e-3,
    epochsPerIteration = 8,
    numberOfCycles = 6,
    totalTrainingPercent = 0.30,
    true_parameters = [100, 0.5], # Create training data for these (p_0, e_0) values
    initial_guess = [100, 0.5],
) # Take this initial (p, e) guesssd
