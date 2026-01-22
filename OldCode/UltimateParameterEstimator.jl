using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/EquationsToWaveform")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/SchwarzschildTrainingData")

function getParameters(solution)
    r_max = maximum(solution[2,:])
    r_min = minimum(solution[2,:])
    e_true = (r_max-r_min)/(r_max+r_min) # True Eccentricity
    p_true = (2*r_max*r_min)/(r_max+r_min) # True semi-latus rectum
    return p_true, e_true
end 

# Global variables for optimization
losses = []
partition_boundaries = []
final_paramaters = []
solutions_list = []
parameters_list = []

function optimizeBlackHole(; learningRate, epochsPerIteration, numberOfCycles, totalTrainingPercent, initial_guess)
    p_guess = initial_guess
    trainingFraction = totalTrainingPercent

    # FIXED: Generate training data ONCE outside the optimization loop
    println("Generating training data...")
    true_problem = create_Schwarzschild_trainingData([10, 0.5])
    true_sol = solve(true_problem, Tsit5(), saveat=240.0)
    
    # Generate true waveforms ONCE
    h_plus_training = compute_waveform(240.0, true_sol, 1.0)[1]
    h_cross_training = compute_waveform(240.0, true_sol, 1.0)[2]
    
    println("Training data generated: $(length(h_plus_training)) points")

    function loss(pn, trainingFraction)
        try
            newprob = remake(true_problem, p = pn)
            sol = solve(newprob, Tsit5(), saveat=240.0)
            
            # Check if solution succeeded
            if sol.retcode != :Success
                println("Warning: Solution failed with retcode: $(sol.retcode)")
                return 1e6  # Large penalty for failed solutions
            end
            
            predicted_waveform_plus = compute_waveform(240.0, sol, 1.0)[1]
            predicted_waveform_cross = compute_waveform(240.0, sol, 1.0)[2]
            
            # Compare only the overlapping portion
            n_train = Int(floor(length(h_plus_training)*trainingFraction))
            n_pred = length(predicted_waveform_plus)
            n_compare = min(n_pred, n_train)
            
            if n_compare == 0
                println("Warning: No points to compare!")
                return 1e6
            end
            
            loss_value = sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
            loss_value += sum(abs2, predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare])
            loss_value /= n_compare
            
            println("Training with fraction: ", trainingFraction, ", n_compare: ", n_compare, ", loss: ", loss_value)
            return loss_value
            
        catch e
            println("Error in loss calculation: $e")
            return 1e6  # Large penalty for errors
        end
    end    
    
    function callback(pn, loss; dotrain = true)
        if dotrain
            push!(losses, loss);
            @printf("Epoch: %d, Loss: %15.12f \n",length(losses),loss);
            p = plot(losses, label = "Loss", xlabel = "Epochs", ylabel = "Loss", 
                    top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
            vline!(partition_boundaries, label = "Partition")
            display(p)
        end
        return false
    end

    # Helper function to handle waveform plotting with proper length matching
    function plot_waveforms(h_plus_pred, h_cross_pred, title_suffix="")
        n_pred = length(h_plus_pred)
        n_train = length(h_plus_training)
        
        # FIXED: Handle all cases for waveform length matching
        if n_pred == n_train
            # Same length - use as is
            h_plus_pred_plot = h_plus_pred
            t_plot = (0:n_train-1) * 240.0
        elseif n_pred < n_train
            # Predicted is shorter - pad with zeros
            h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
            t_plot = (0:n_train-1) * 240.0
        else  # n_pred > n_train
            # Predicted is longer - truncate
            h_plus_pred_plot = h_plus_pred[1:n_train]
            t_plot = (0:n_train-1) * 240.0
        end

        p = plot(t_plot, h_plus_training, label="h+ true", linewidth=2,
                xlabel="Time (s)", ylabel="h+ Amplitude",
                legend=:topright, grid=true, 
                top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm,
                title="Waveform Comparison $title_suffix")
        plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)
        display(p)
        
        return h_plus_pred_plot
    end

    # define Optimization Problem
    adtype = Optimization.AutoFiniteDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_guess)

    # choose method for solving problem 
    lr = learningRate;
    opt_method = Optimisers.Adam(lr)

    # solve the problem
    num_iters = epochsPerIteration;
    println("Starting initial optimization...")
    opt_result = Optimization.solve(optprob, opt_method, callback=callback, maxiters=num_iters)
    p_final = opt_result.u

    # Generate and plot initial result
    newprob = remake(true_problem, p = p_final)
    sol = solve(newprob, Tsit5(), saveat=240.0)
    push!(solutions_list, sol)
    h_plus_pred = compute_waveform(240.0, sol, 1.0)[1]
    h_cross_pred = compute_waveform(240.0, sol, 1.0)[2]

    plot_waveforms(h_plus_pred, h_cross_pred, "(Initial)")

    function partitionTraining(numCycles, totalTrainingFraction)
        amountTrain = totalTrainingFraction / numCycles
        p_final_array = [p_final]
    
        for i in 1:numCycles
            println("Starting partition $i of $numCycles...")
            trainingFraction_current = i * amountTrain
            push!(partition_boundaries, length(losses))
            push!(final_paramaters, p_final_array[end])
            
            optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction_current), adtype)
            optprob = Optimization.OptimizationProblem(optf, p_final_array[end])
            opt_result_2 = Optimization.solve(optprob, opt_method, callback = callback, maxiters = num_iters)
    
            p_final_2 = opt_result_2.u
            push!(p_final_array, p_final_2)
            
            newprob_2 = remake(true_problem, p = p_final_2)
            sol_2 = solve(newprob_2, Tsit5(), saveat=240.0)
            push!(solutions_list, sol_2)
            push!(parameters_list, getParameters(sol_2))
    
            h_plus_pred = compute_waveform(240.0, sol_2, 1.0)[1]
            h_cross_pred = compute_waveform(240.0, sol_2, 1.0)[2]

            plot_waveforms(h_plus_pred, h_cross_pred, "(Cycle $i)")
        end
    end    

    numCycles = numberOfCycles
    partitionTraining(numCycles, trainingFraction)

    # Final parameter visualization
    x = range(6, 12, length=20)
    y = (x .- 6) ./ 2
    p = plot(x, y, ylims=(-0.1, 1), xlims = (6, 12), linewidth = 3, 
            bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm, 
            label = "Separatrix", xlabel = "p (Semi-latus Rectum)", ylabel = "e (Eccentricity)", 
            legend=:bottomright, title="Parameter Estimation Results")
    
    true_params = getParameters(true_sol)
    scatter!([true_params[1]], [true_params[2]], color = "lightsalmon", markersize = 5, label = "True Parameters")
    
    for i in 1:numCycles
        scatter!([parameters_list[i][1]], [parameters_list[i][2]], color = "darkseagreen1", markersize = 3, legend = false)
    end
    
    display(p)

    final_error = (parameters_list[end][1] - true_params[1])^2 + (parameters_list[end][2] - true_params[2])^2
    println("Final parameter error: $final_error")
    println("True parameters: p=$(true_params[1]), e=$(true_params[2])")
    println("Final estimated parameters: p=$(parameters_list[end][1]), e=$(parameters_list[end][2])")
    
    return final_error
end

# Clear global variables before running
global losses = []
global partition_boundaries = []
global final_paramaters = []
global solutions_list = []
global parameters_list = []

# Run optimization
result = optimizeBlackHole(learningRate = 5e-3, 
                          epochsPerIteration = 25, 
                          numberOfCycles = 3, 
                          totalTrainingPercent = 0.05, 
                          initial_guess = [1, 0.95, 3.5])