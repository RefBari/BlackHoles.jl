using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures

# include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ParameterEsimationFinally!")

# Modified optimizeBlackHole function for silent operation
function optimizeBlackHole_silent(; learningRate, epochsPerIteration, totalTrainingPercent, initial_guess)
    # Reset global variables to avoid accumulation
    local_losses = []
    
    p_guess = initial_guess
    trainingFraction = totalTrainingPercent

    function loss(pn, trainingFraction)
        newprob = remake(prob, p = pn)
        sol = solve(newprob, Tsit5(), saveat=0.1)
        predicted_waveform_plus = compute_waveform(0.1, sol, 1.0)[1]
        predicted_waveform_cross = compute_waveform(0.1, sol, 1.0)[2]
        
        # Compare only the overlapping portion
        n_train = Int(floor(length(h_plus_training)*trainingFraction))
        n_pred = length(predicted_waveform_plus)
        n_compare = min(n_pred, n_train)
        
        loss_value = sum(abs2, predicted_waveform_plus[1:n_compare] .- h_plus_training[1:n_compare])
        loss_value += sum(abs2, predicted_waveform_cross[1:n_compare] .- h_cross_training[1:n_compare])
        loss_value /= n_compare
        return loss_value
    end    
    
    # Silent callback - no plotting
    function silent_callback(pn, loss_val)
        push!(local_losses, loss_val)
        return false
    end

    try
        # Define optimization problem
        adtype = Optimization.AutoFiniteDiff()
        optf = Optimization.OptimizationFunction((x, p) -> loss(x, trainingFraction), adtype)
        optprob = Optimization.OptimizationProblem(optf, p_guess)

        # Solve the problem
        lr = learningRate
        opt_method = Optimisers.Adam(lr)
        opt_result = Optimization.solve(optprob, opt_method, callback=silent_callback, maxiters=epochsPerIteration)
        
        return local_losses[end]  # Return final loss
    catch e
        println("Optimization failed: $e")
        return Inf  # Return large loss if optimization fails
    end
end

# Domain of Convergence analysis function
function find_domain_of_convergence(p_center, e_center; 
                                   tolerance=1e-4, 
                                   max_radius=1.0, 
                                   radius_step=0.1, 
                                   n_angles=8,
                                   learningRate=5e-3,
                                   epochsPerIteration=25,
                                   totalTrainingPercent=0.05)
    
    println("Finding DoC for center point ($p_center, $e_center)")
    
    current_radius = radius_step
    doc_radius = 0.0
    
    while current_radius <= max_radius
        println("  Testing radius: $current_radius")
        
        # Generate points on circumference
        angles = range(0, 2π, length=n_angles+1)[1:end-1]
        p_samples = p_center .+ current_radius .* cos.(angles)
        e_samples = e_center .+ current_radius .* sin.(angles)
        
        losses_at_radius = []
        valid_points = 0
        
        # Test each point on circumference
        for i in 1:length(p_samples)
            # Check if point is physically valid
            if p_samples[i] > 0 && e_samples[i] > 0 && e_samples[i] < (p_samples[i] - 6) / 2
                valid_points += 1
                
                # Convert (p,e) to (E,L) parameters
                E_value, L_value = pe_2_EL(p_samples[i], e_samples[i])[2:3]
                
                # Run optimization
                final_loss = optimizeBlackHole_silent(
                    learningRate=learningRate,
                    epochsPerIteration=epochsPerIteration,
                    totalTrainingPercent=totalTrainingPercent,
                    initial_guess=[1, E_value, L_value]
                )
                
                push!(losses_at_radius, final_loss)
                println("    Point ($(@sprintf("%.2f", p_samples[i])), $(@sprintf("%.2f", e_samples[i]))): loss = $(@sprintf("%.2e", final_loss))")
            end
        end
        
        # Check convergence condition
        if valid_points == 0
            println("  No valid points at radius $current_radius")
            break
        elseif all(loss < tolerance for loss in losses_at_radius)
            # All losses below tolerance - this radius is within DoC
            doc_radius = current_radius
            println("  ✓ All losses below tolerance at radius $current_radius")
            current_radius += radius_step
        else
            # At least one loss above tolerance - DoC boundary found
            println("  ✗ Some losses above tolerance at radius $current_radius")
            println("  Domain of Convergence radius: $doc_radius")
            break
        end
    end
    
    return doc_radius
end

# Main stress test function
function run_doc_stress_test()
    # Grid parameters
    p_sampler = 6:1:12
    e_sampler = 0:0.1:1
    
    # DoC analysis parameters
    doc_results = Dict()
    
    # Initialize plot
    x = range(6, 12, length=20)
    y = (x .- 6) ./ 2
    p = plot(x, y, linewidth=3, bottom_margin=10mm, top_margin=10mm, 
             left_margin=10mm, right_margin=10mm, label="Separatrix", 
             xlabel="p (Semi-latus Rectum)", ylabel="e (Eccentricity)", 
             legend=:topright, aspect_ratio=:equal)
    
    # Run DoC analysis for each valid grid point
    for p_val in p_sampler
        for e_val in e_sampler
            if e_val < (p_val - 6) / 2  # Check if below separatrix (bound orbit)
                
                # Find domain of convergence
                doc_radius = find_domain_of_convergence(p_val, e_val)
                doc_results[(p_val, e_val)] = doc_radius
                
                # Plot center point
                scatter!([p_val], [e_val], color="red", markersize=5, legend=false)
                
                # Plot DoC circle
                if doc_radius > 0
                    theta_smooth = range(0, 2π, length=100)
                    p_smooth = p_val .+ doc_radius .* cos.(theta_smooth)
                    e_smooth = e_val .+ doc_radius .* sin.(theta_smooth)
                    plot!(p_smooth, e_smooth, color=:green, alpha=0.6, linewidth=2, label="DoC")
                end
                
                println("DoC for ($p_val, $e_val): $doc_radius")
            end
        end
    end
    
    display(p)
    return doc_results
end

# Usage:
doc_results = run_doc_stress_test()