function optimizeBlackHole(lr, epochs, numCycles, totalTrainingPercent, p_0)
    # global prob, h_plus_training, h_cross_training, true_sol, R, M    
    function getParameters(solution)
        r_max = maximum(solution[2,:])
        r_min = minimum(solution[2,:])
        e_true = (r_max-r_min)/(r_max+r_min) # True Eccentricity
        p_true = (2*r_min*r_max)/(r_min+r_max) # True semi-latus rectum
        return p_true, e_true
    end 

    p_true = [1.0, 
              Circular_Orbit_Energy(R, M), 
              Angular_Momentum(R, M)]

    p_guess = p_0

    function loss(pn, trainingFraction)
        newprob = remake(prob, p = pn)
        sol = solve(newprob, Tsit5(), saveat=0.1)
        predicted_waveform_plus = compute_waveform(0.1, sol, 1.0)[1]
        predicted_waveform_cross = compute_waveform(0.1, sol, 1.0)[2]
        
        # Compare only the overlapping portion
        n_train = Int(floor(length(h_plus_training)*trainingFraction))
        n_pred = length(predicted_waveform_plus)
        n_compare = min(n_pred, n_train)
        
        loss = sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
        loss += sum(abs2, predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare])
        println("Training with fraction: ", trainingFraction, ", n_compare: ", n_compare, ", loss: ", loss)
        return loss
    end

    ### STEP 1 of 5 FOR PARTITIONED TRAINING PROCESS

    trainingFraction = 0.01

    losses = []
    partition_boundaries = []
    final_paramaters = []
    solutions_list = []
    parameters_list = []

    function callback(pn, loss_val; dotrain = true)
    if dotrain
        push!(losses, loss_val);
        @printf("Epoch: %d, Loss: %15.12f \n",length(losses),loss_val);
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
    opt_method = Optimisers.Adam(lr)

    # solve the problem
    num_iters = epochs; # number of iterations per partition

    opt_result = Optimization.solve(optprob, opt_method, callback=callback, maxiters=num_iters)

    p_final = opt_result.u

    newprob = remake(prob, p = p_final)
    sol = solve(newprob, Tsit5(), saveat=0.1)
    push!(solutions_list, sol)
    h_plus_pred = compute_waveform(0.1, sol, 1.0)[1]
    h_cross_pred = compute_waveform(0.1, sol, 1.0)[2]

    # Handle all cases properly
    n_pred = length(h_plus_pred)
    n_train = length(h_plus_training)

    if n_pred == n_train
        # Same length - no padding needed
        h_plus_pred_plot = h_plus_pred
    elseif n_pred < n_train
        # Predicted is shorter - pad with zeros
        h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
    end

    t_plot = (0:n_train-1) * 0.1

    plot(t_plot, h_plus_training, label="h+ true", linewidth=2,
         xlabel="Time (s)", ylabel="h+ Amplitude",
         legend=:topright, grid=true, top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
    plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)

    function partitionTraining(numCycles, totalTrainingFraction)
        
        amountTrain = totalTrainingFraction / numCycles
        p_final_array = [p_final]

        for i in 1:numCycles
            trainingFraction = i * amountTrain
            push!(partition_boundaries, length(losses))
            push!(final_paramaters, p_final_array[end])
            optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
            optprob = Optimization.OptimizationProblem(optf, p_final_array[end])
            opt_result_2 = Optimization.solve(optprob, opt_method, callback = callback, maxiters = num_iters)

            p_final_2 = opt_result_2.u
            push!(p_final_array, p_final_2)
            newprob_2 = remake(prob, p = p_final_2)
            sol_2 = solve(newprob_2, Tsit5(), saveat=0.1)
            push!(solutions_list, sol_2)
            push!(parameters_list, getParameters(sol_2))

            h_plus_pred = compute_waveform(0.1, sol_2, 1.0)[1]
            h_cross_pred = compute_waveform(0.1, sol_2, 1.0)[2]

            # Handle all cases properly
            n_pred = length(h_plus_pred)
            n_train = length(h_plus_training)

            if n_pred == n_train
                # Same length - no padding needed
                h_plus_pred_plot = h_plus_pred
            elseif n_pred < n_train
                # Predicted is shorter - pad with zeros
                h_plus_pred_plot = [h_plus_pred; zeros(n_train - n_pred)]
            end
            
            t_plot = (0:n_train-1) * 0.1

            p = plot(t_plot, h_plus_training, label="h+ true", linewidth=2,
            xlabel="Time (s)", ylabel="h+ Amplitude",
            legend=:topright, grid=true, bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
            plot!(t_plot, h_plus_pred_plot, label="h+ predicted", linewidth=2)
            display(p)
        end
        
        return p_final_array
    end

    # Run partition training
    final_parameters = partitionTraining(numCycles, totalTrainingPercent)

    # Create final plot
    x = range(6, 12, length=20)
    y = (x .- 6) ./ 2
    p = plot(x, y, ylims=(-0.1, 1), xlims = (6, 12), bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm, label = "Separatrix", xlabel = "p", ylabel = "e")
    scatter!([getParameters(true_sol)[1]], [getParameters(true_sol)[2]], color = "lightsalmon", markersize = 5, label = "True Parameters")
    for i in 1:numCycles
        scatter!([parameters_list[i][1]], [parameters_list[i][2]], color = "darkseagreen1", markersize = 3, label = "Iteration $i")
    end
    display(p)
    
    # Return useful results
    return (
        final_parameters = final_parameters,
        solutions = solutions_list,
        parameters = parameters_list,
        losses = losses
    )
end

# Example usage:
results = optimizeBlackHole(1e-2, 25, 2, 0.01, [1.0, 0.94, 3.5])