using Plots

# Create a clean example of concentric circles
function plot_concentric_circles()
    # Set up the plot
    p = plot(xlabel="p (Semi-latus rectum)", ylabel="e (Eccentricity)", 
             xlims=(6, 12), ylims=(-0.1, 1),
             title="Clean Concentric Circles", legend=false,
             aspect_ratio=:equal)
    
    # Plot separatrix
    x = range(6, 12, length=20)
    y = (x .- 6) ./ 2
    plot!(x, y, linewidth=3, color=:red, label="Separatrix")
    
    # Define grid points and sampling parameters
    p_centers = [8, 10]  # Just 2 points for clarity
    e_centers = [0.3, 0.6]
    
    radii = [0.3, 0.6]  # Two concentric circles
    n_points = 8        # 8 points per circle
    
    # Generate circles for each center point
    for p0 in p_centers
        for e0 in e_centers
            # Check if point is above separatrix (bound orbit)
            if e0 > (p0 - 6) / 2
                
                # Plot center point
                scatter!([p0], [e0], color=:black, markersize=6)
                
                # Generate each circle
                for (i, radius) in enumerate(radii)
                    # Create angles for this circle
                    angles = range(0, 2π, length=n_points+1)[1:end-1]
                    
                    # Calculate circle points
                    p_circle = p0 .+ radius .* cos.(angles)
                    e_circle = e0 .+ radius .* sin.(angles)
                    
                    # Choose color for this radius
                    colors = [:blue, :green, :orange]
                    circle_color = colors[i]
                    
                    # Plot points on circle circumference
                    scatter!(p_circle, e_circle, color=circle_color, markersize=3)
                end
            end
        end
    end
    
    return p
end

# Test with different numbers of points
function test_different_point_counts()
    fig = plot(layout=(1,3), size=(900, 300))
    
    p0, e0 = 9.0, 0.4
    radius = 0.5
    
    for (subplot, n_pts) in enumerate([4, 8, 16])
        # Calculate angles
        angles = range(0, 2π, length=n_pts+1)[1:end-1]
        
        # Calculate circle points
        p_circle = p0 .+ radius .* cos.(angles)
        e_circle = e0 .+ radius .* sin.(angles)
        
        # Plot
        scatter!(p_circle, e_circle, subplot=subplot, 
                title="$n_pts points", markersize=4, color=:blue,
                aspect_ratio=:equal)
        scatter!([p0], [e0], subplot=subplot, color=:red, markersize=6)
        
        # Draw the actual circle for reference
        theta_smooth = range(0, 2π, length=100)
        p_smooth = p0 .+ radius .* cos.(theta_smooth)
        e_smooth = e0 .+ radius .* sin.(theta_smooth)
        plot!(p_smooth, e_smooth, subplot=subplot, color=:gray, alpha=0.3, linewidth=1)
    end
    
    return fig
end

# Run the examples
p1 = plot_concentric_circles()
p2 = test_different_point_counts()

display(p1)
display(p2)