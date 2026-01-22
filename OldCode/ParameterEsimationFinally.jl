using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/SchwarzschildTrainingData")

losses = []
partition_boundaries = []
final_paramaters = []
solutions_list = []
parameters_list = []
final_predicted_h_plus = []
training_h_plus_wave = []

function optimizeBlackHole(; learningRate, epochsPerIteration, numberOfCycles, totalTrainingPercent, true_parameters, initial_guess)
    global losses = []
    global partition_boundaries = []
    global final_paramaters = []
    global solutions_list = []
    global parameters_list = []
    global final_predicted_h_plus = []
    global training_h_plus_wave = []

    p_guess = pe_2_EL(initial_guess[1], initial_guess[2])
    print("Using pe_2_EL, we find [M, E, L] is", p_guess)
    trainingFraction = totalTrainingPercent
    
    true_p = true_parameters[1]
    true_e = true_parameters[2]

    trainingData = create_Schwarzschild_trainingData([true_p, true_e])

    timestep = 1
    true_problem = trainingData[1]
    true_solution = trainingData[2]

    function loss(pn, trainingFraction)
        newprob = remake(true_problem, p = pn)
        sol = solve(newprob, Tsit5(), saveat=timestep)
        
        predicted_waveform_plus = compute_waveform(timestep, sol, 1.0)[1]
        predicted_waveform_cross = compute_waveform(timestep, sol, 1.0)[2]
        
        h_plus_training = trainingData[3]
        h_cross_training = trainingData[4]

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
            p = plot(losses, label = "Loss", xlabel = "Epochs", ylabel = "Loss", top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
            vline!(partition_boundaries, label = "Partition")
            display(p)
        end
        return false
    end

    # define Optimization Problem
    adtype = Optimization.AutoFiniteDiff() # instead of Optimization.AutoZygote(), use finite differences
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
    
    # create Optimization Problem (function + initial guess for parameters)
    optprob = Optimization.OptimizationProblem(optf, p_guess)

    # choose method for solving problem 
    lr = learningRate;

    # solve the problem
    num_iters = epochsPerIteration; # number of iterations per partition (i.e., 2 partitions means one run + 2 additional runs = 3 runs * 25 epochs/run = 75 epochs)
    opt_result = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = lr), callback=callback, maxiters=num_iters)
    p_final = opt_result.u

    newprob = remake(true_problem, p = p_final)
    sol = solve(newprob, Tsit5(), saveat=timestep)
    push!(solutions_list, sol)

    h_plus_pred = compute_waveform(timestep, sol, 1.0)[1]
    h_cross_pred = compute_waveform(timestep, sol, 1.0)[2]

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

    t_plot = (0:n_train-1) * timestep

    p = plot(t_plot, h_plus_training, label="h+ true", linewidth=2,
        xlabel="Time (s)", ylabel="h+ Amplitude",
        legend=:topright, grid=true, top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
    plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)# plot!(t_plot, predicted_waveform_plus_old_padded, label="h+ initial prediction", linewidth=2)
    display(p)

    function partitionTraining(numCycles, totalTrainingFraction)
        global partition_boundaries, losses, final_paramaters, solutions_list, parameters_list
        
        amountTrain = totalTrainingFraction / numCycles
        p_final_array = [p_final]
    
        for i in 1:numCycles
            trainingFraction = i * amountTrain
            push!(partition_boundaries, length(losses))
            push!(final_paramaters, p_final_array[end])
            optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
            optprob = Optimization.OptimizationProblem(optf, p_final_array[end])
            opt_result_2 = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = lr), callback=callback, maxiters=num_iters)
    
            p_final_2 = opt_result_2.u
            push!(p_final_array, p_final_2)
            newprob_2 = remake(true_problem, p = p_final_2)
            sol_2 = solve(newprob_2, Tsit5(), saveat=timestep)
            push!(solutions_list, sol_2)
            push!(parameters_list, getParameters(sol_2))
    
            h_plus_pred = compute_waveform(timestep, sol_2, 1.0)[1]
            h_cross_pred = compute_waveform(timestep, sol_2, 1.0)[2]
    
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
            
            t_plot = (0:n_train-1) * timestep
    
            p = plot(t_plot, h_plus_training, color = "lightsalmon", label="h+ true", linewidth=2,
            xlabel="Time (s)", ylabel="h+ Amplitude",
            legend=:topright, grid=true, bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
            plot!(t_plot, h_plus_pred_plot, label="h+ predicted",
                    markershape=:circle, markeralpha = 0.20,
                    linewidth = 2, alpha = 0.25, linestyle=:dash)# plot!(t_plot, predicted_waveform_plus_old_padded, label="h+ initial prediction", linewidth=2)
            display(p)
            push!(p, h_plus_pred_plot)
            push!(training_h_plus_wave, h_plus_training)
        end
    end    

    numCycles = numberOfCycles

    partitionTraining(numCycles, trainingFraction)

    x = range(6, 12, length=20)
    y = (x .- 6) ./ 2
    p = plot(x, y, ylims=(-0.1, 1), xlims = (6, 12), linewidth = 3, bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm, label = "Separatrix", xlabel = "p (Semi-latus Rectum)", ylabel = "e (Eccentricity)", legend=:bottomright)
    scatter!([getParameters(true_solution)[1]], [getParameters(true_solution)[2]], color = "lightsalmon", markersize = 5, label = "True Parameters")
    
    for i in 1:numCycles
        scatter!([parameters_list[i][1]], [parameters_list[i][2]], color = "darkseagreen1", markersize = 3, legend = false)
    end
    
    display(p)

    return (parameters_list[end][1] - getParameters(true_solution)[1])^2 + (parameters_list[end][2] - getParameters(true_solution)[2])^2
end

optimizeBlackHole(learningRate = 1e-3, 
                  epochsPerIteration = 3, 
                  numberOfCycles = 2, 
                  totalTrainingPercent = 0.1, 
                  true_parameters = [100, 0.5], # Create training data for these (p_0, e_0) values
                  initial_guess = [101, 0.6]) # Take this initial (p, e) guess