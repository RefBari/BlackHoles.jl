using Plots
using DifferentialEquations
using LaTeXStrings
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures
using OrdinaryDiffEq, DiffEqFlux
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux, ComponentArrays
using Printf
using Random

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/SchwarzschildTrainingData")

losses = []
partition_boundaries = []
final_paramaters = []
solutions_list = []
parameters_list = []
final_predicted_h_plus = []
training_h_plus_wave = []

function optimizeBlackHole(; learningRate, epochsPerIteration, numberOfCycles, totalTrainingPercent, true_parameters, initial_guess)
    # Initialize tracking arrays
    global losses, partition_boundaries, parameters_list
    losses = []
    partition_boundaries = []
    parameters_list = []
    
    p_guess = pe_2_EL(initial_guess[1], initial_guess[2])
    trainingFraction = totalTrainingPercent
    
    true_p = true_parameters[1]
    true_e = true_parameters[2]

    trainingData = create_Schwarzschild_trainingData([true_p, true_e])

    true_problem = trainingData[1]
    true_solution = trainingData[2]
    h_plus_training = trainingData[3]
    h_cross_training = trainingData[4]

    # Define u0 and tspan from the training data
    u0 = true_problem.u0
    tspan = true_problem.tspan

    timestep = 1
    
    # Create the neural network that will learn corrections to the flat space equations of motion and learn the geodesic equations of motion
    NN = Chain(Dense(8, 4,tanh),
            Dense(4, 4, tanh),
            Dense(4, 3))
    rng = MersenneTwister(222)
    NN_params, NN_state = Lux.setup(rng, NN)
    precision = Float64
    NN_params = Lux.fmap(x -> precision.(x), NN_params)

    # Make the weights and biases of the neural network zero and add tiny random values
    for layer in NN_params
        if ~isempty(layer)
            layer.weight .*= 0 .* layer.weight .+ precision(1e-6) * (randn(rng, eltype(layer.weight), size(layer.weight)))
            layer.bias .*= 0 .* layer.bias .+ precision(1e-6) * (randn(rng, eltype(layer.bias), size(layer.bias)))
        end
    end

    θ = (; M = p_guess[1], E = p_guess[2], L = p_guess[3], NN = NN_params)
    θ = ComponentVector{precision}(θ);

    function SchwarzschildNN(du, u, p, t)
        coord_time, r, θ, ϕ, p_t, p_r, p_θ, p_ϕ = u
        M = p.M
        E = p.E
        L = p.L

        NN_params = p.NN
        F = NN(u, NN_params, NN_state)[1]

        du[1] = dt = 1 * (1+1e-3*F[1])
        du[2] = dr = p_r * (1+1e-3*F[2])
        du[3] = dθ = 0
        du[4] = dϕ = L/r^2

        du[5] = dp_t = 0
        du[6] = dp_r = (-M/r^2 + L^2/(r^3))*(1+1e-3*F[3])
        du[7] = dp_θ = 0
        du[8] = dp_ϕ = 0
    end

    prob_learn = ODEProblem(SchwarzschildNN, u0, tspan, θ)

    function loss(pn, trainingFraction)
        newprob = remake(prob_learn, p = pn)
        sol = solve(newprob, Tsit5(), saveat=timestep)
        
        predicted_waveform_plus = compute_waveform(timestep, sol, 1.0)[1]
        predicted_waveform_cross = compute_waveform(timestep, sol, 1.0)[2]

        # Compare only the overlapping portion
        n_train = Int(floor(length(h_plus_training)*trainingFraction))
        n_pred = length(predicted_waveform_plus)
        n_compare = min(n_pred, n_train)
        
        loss_value = sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
        loss_value += sum(abs2, predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare])
        loss_value /= n_compare
        println("Training with fraction: ", trainingFraction, ", n_compare: ", n_compare, ", loss: ", loss_value)
        return loss_value
    end
    
    function callback(pn, loss; dotrain = true)
        if dotrain
            push!(losses, loss);
            @printf("Epoch: %d, Loss: %15.12f \n",length(losses),loss);
            
            # Create all three plots in real-time
            try
                # Get current solution and parameters
                current_prob = remake(prob_learn, p = pn)
                current_sol = solve(current_prob, Tsit5(), saveat=timestep)
                current_h_plus = compute_waveform(timestep, current_sol, 1.0)[1]
                current_params = getParameters(current_sol)
                
                # Plot 1: Loss function
                p1 = plot(losses, label = "Loss", xlabel = "Epochs", ylabel = "Loss", 
                          title = "Training Loss",
                          top_margin = 3mm, bottom_margin = 3mm, left_margin = 3mm, right_margin = 3mm)
                vline!(p1, partition_boundaries, label = "Partition", alpha=0.7)
                
                # Plot 2: Gravitational wave fit  
                n_pred = length(current_h_plus)
                n_train = length(h_plus_training)
                n_compare = min(n_pred, n_train, 2000)  # Limit to 2000 points for speed
                
                t_plot_short = (0:n_compare-1) * timestep
                p2 = plot(t_plot_short, h_plus_training[1:n_compare], label="h+ true", linewidth=2,
                    xlabel="Time", ylabel="h+ Amplitude", title="Waveform Fit",
                    top_margin = 3mm, bottom_margin = 3mm, left_margin = 3mm, right_margin = 3mm)
                plot!(p2, t_plot_short, current_h_plus[1:n_compare], label="h+ predicted", linewidth=2)
                
                # Plot 3: p-e phase space with trajectory
                x = range(6, 12, length=20)
                y = (x .- 6) ./ 2
                p3 = plot(x, y, label = "Separatrix", xlabel = "p", ylabel = "e", 
                         title="Parameter Space", linewidth = 2,
                         ylims=(-0.1, 1), xlims = (6, 12),
                         top_margin = 3mm, bottom_margin = 3mm, left_margin = 3mm, right_margin = 3mm)
                
                # True parameters
                true_params = getParameters(true_solution)
                scatter!(p3, [true_params[1]], [true_params[2]], 
                        color = "red", markersize = 6, label = "True")
                
                # Current parameters
                scatter!(p3, [current_params[1]], [current_params[2]], 
                        color = "blue", markersize = 6, label = "Current")
                
                # Show trajectory if we have multiple points
                if length(parameters_list) > 0
                    param_p = [p[1] for p in parameters_list]
                    param_e = [p[2] for p in parameters_list]
                    plot!(p3, param_p, param_e, label="Trajectory", alpha=0.7, linewidth=1)
                end
                
                # Combine all three plots
                l = @layout [a b c]  # Side-by-side horizontal layout
                combined_plot = plot(p1, p2, p3, layout = l, size=(1400, 400))
                display(combined_plot)
                
            catch e
                println("Error in callback plotting: ", e)
                # Just show loss plot if there's an error
                p1 = plot(losses, label = "Loss", xlabel = "Epochs", ylabel = "Loss", title = "Training Loss")
                vline!(p1, partition_boundaries, label = "Partition")
                display(p1)
            end
        end
        return false
    end

    # define Optimization Problem
    adtype = Optimization.AutoFiniteDiff() # instead of Optimization.AutoZygote(), use finite differences
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
    
    # create Optimization Problem (function + initial guess for parameters)
    optprob = Optimization.OptimizationProblem(optf, θ)

    # choose method for solving problem 
    lr = learningRate;

    # solve the problem
    num_iters = epochsPerIteration; # number of iterations per partition (i.e., 2 partitions means one run + 2 additional runs = 3 runs * 25 epochs/run = 75 epochs)
    opt_result = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = lr), callback=callback, maxiters=num_iters)
    p_final = opt_result.u

    newprob = remake(prob_learn, p = p_final)
    sol = solve(newprob, Tsit5(), saveat=timestep)
    push!(solutions_list, sol)

    h_plus_pred = compute_waveform(timestep, sol, 1.0)[1]
    h_cross_pred = compute_waveform(timestep, sol, 1.0)[2]

    # Handle all cases properly
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

    t_plot = (0:n_train-1) * timestep

    p = plot(t_plot, h_plus_training, label="h+ true", linewidth=2,
        xlabel="Time (s)", ylabel="h+ Amplitude",
        legend=:topright, grid=true, top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
    plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)
    display(p)

    function partitionTraining(numCycles, totalTrainingFraction)
        global partition_boundaries, losses, final_paramaters, solutions_list, parameters_list, final_predicted_h_plus
        
        amountTrain = totalTrainingFraction / numCycles
        p_final_array = [p_final]
    
        for i in 1:numCycles
            trainingFraction = i * amountTrain
            push!(partition_boundaries, length(losses))
            push!(final_paramaters, p_final_array[end])
            optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
            θ_current = ComponentVector{precision}((; M = p_final_array[end][1], E = p_final_array[end][2], L = p_final_array[end][3], NN = NN_params))
            optprob = Optimization.OptimizationProblem(optf, θ_current)
            opt_result_2 = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = lr), callback=callback, maxiters=num_iters)
    
            p_final_2 = opt_result_2.u
            push!(p_final_array, p_final_2)
            newprob_2 = remake(prob_learn, p = p_final_2)
            sol_2 = solve(newprob_2, Tsit5(), saveat=timestep)
            push!(solutions_list, sol_2)
            push!(parameters_list, getParameters(sol_2))
    
            h_plus_pred = compute_waveform(timestep, sol_2, 1.0)[1]
            h_cross_pred = compute_waveform(timestep, sol_2, 1.0)[2]
            
            # Save the predicted waveform
            n_pred = length(h_plus_pred)
            n_train = length(h_plus_training)
    
            if n_pred == n_train
                h_plus_pred_plot = h_plus_pred
            elseif n_pred < n_train
                h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
            else
                h_plus_pred_plot = h_plus_pred[1:n_train]
            end
            
            push!(final_predicted_h_plus, h_plus_pred_plot)
            push!(training_h_plus_wave, h_plus_training)
        end
    end    

    numCycles = numberOfCycles
    partitionTraining(numCycles, trainingFraction)

    # Return final error
    return (parameters_list[end][1] - getParameters(true_solution)[1])^2 + (parameters_list[end][2] - getParameters(true_solution)[2])^2
end

optimizeBlackHole(learningRate = 1e-3, 
                  epochsPerIteration = 3, 
                  numberOfCycles = 2, 
                  totalTrainingPercent = 0.1, 
                  true_parameters = [10, 0.1], # Create training data for these (p_0, e_0) values
                  initial_guess = [10.2, 0.2]) # Take this initial (p, e) guess