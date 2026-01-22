using Plots
using DifferentialEquations
using LaTeXStrings
using Optimization, OptimizationOptimJL
using Optimisers
using OptimizationOptimisers
using Printf
using Plots.Measures
using Base.Iterators

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/ParameterEsimationFinally!")

radius = 0.15
num_points = 2
p_sampler = 6:2:12
e_sampler = 0:0.6:1
pe_sampling_values = [(7.5, 0.2), (12, 0.2), (9, 0.8), (13.5, 0.8)]
num_samples = length(pe_sampling_values)
final_errors = []
sampling_points = []

function create_sampling_circle(size, numberPoints, p0, e0)
    r = size
    angs = (0:pi/numberPoints:2*pi)[1:end-1]

    for θ in angs
        scatter!([p0 + r*cos(θ)], [e0 + r*sin(θ)], 
                  markercolor = "lightblue", markersize = 2)
        push!(sampling_points, (6*(p0 + r*cos(θ)), e0 + r*sin(θ)))
    end
end

function find_doc_radius(p0, e0; initial_radius=0.05, radius_step=0.05, 
                        max_radius=0.5, tolerance=1e-4, num_points=2)
    
    current_radius = initial_radius
    doc_radius = 0.0
    
    while current_radius <= max_radius
        # Clear the sampling points and create new circle
        global sampling_points = []
        create_sampling_circle(current_radius, num_points, p0/6, e0)
        
        # Test all points on this circle
        all_converged = true
        
        for point in sampling_points
            error = optimizeBlackHole(
                learningRate = 1e-3, 
                epochsPerIteration = 3, 
                numberOfCycles = 2, 
                totalTrainingPercent = 0.1, 
                true_parameters = [p0, e0],
                initial_guess = point
            )
            
            if error >= tolerance
                all_converged = false
                println("  Point $point failed with error $error")
                break  # Stop testing this radius
            else
                println("  Point $point passed with error $error")
            end
        end
        
        if all_converged
            # All points converged - this radius is good
            doc_radius = current_radius
            println("  ✓ All points converged at radius $current_radius")
            current_radius += radius_step
        else
            # At least one point failed - stop here
            println("  ✗ DoC boundary found. Final DoC radius: $doc_radius")
            break
        end
    end
    
    return doc_radius
end

# Find DoC for all points
doc_radii = []
for (p0, e0) in pe_sampling_values
    doc_radius = find_doc_radius(p0, e0, initial_radius=0.01, radius_step=0.01)
    push!(doc_radii, doc_radius)
end

# doc_radius_1 = find_doc_radius(9.0, 0.8, initial_radius=0.01, radius_step=0.01)
# doc_radius_2 = find_doc_radius(9.0, 0.8, initial_radius=0.01, radius_step=0.01)
# doc_radius_3 = find_doc_radius(9.0, 0.8, initial_radius=0.01, radius_step=0.01)
# doc_radius_4 = find_doc_radius(9.0, 0.8, initial_radius=0.01, radius_step=0.01)

x = range(6, 12.5, length=20)
y = (x .- 6) ./ 2
q = plot(x ./ 6, y, linewidth = 3, bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm, label = "Separatrix", xlabel = "p/6 (Semilatus Rectum)", ylabel = "e (Eccentricity)", xlims = (1, 2.5), ylims = (0, 1), legend = false)

for i in pe_sampling_values
    scatter!([i[1] / 6], [i[2]], markercolor = "red", markersize = 3)
    # create_sampling_circle(radius, num_points/2, i[1] / 6, i[2])
end

grouped_points = collect(Iterators.partition(sampling_points, num_points))
counter = length(grouped_points)

# Plot DoC circles (match your coordinate system exactly)
for (i, (p0, e0)) in enumerate(pe_sampling_values)
    theta_smooth = range(0, 2π, length=100)
    p_smooth = (p0/6) .+ doc_radii[i] .* cos.(theta_smooth)
    e_smooth = e0 .+ doc_radii[i] .* sin.(theta_smooth)
    plot!(p_smooth, e_smooth, color=:green, alpha=0.7, linewidth=2)
end

display(q)
# display(q)

# for i in 1:num_samples
#     for j in 1:2
        
#         if error > 1e-4
#             print("Sad! Error > 1e-4")
#         else
#             print("Happy! Error < 1e-4")
#         end
#         push!(final_errors, error)
#     end
# end

error = optimizeBlackHole(learningRate = 1e-3, 
                        epochsPerIteration = 3, 
                        numberOfCycles = 2, 
                        totalTrainingPercent = 0.1, 
                        true_parameters = [pe_sampling_values[3][1], pe_sampling_values[3][2]], # Create training data for these (p_0, e_0) values
                        initial_guess = grouped_points[3][2])