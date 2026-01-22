using DifferentialEquations, Flux, DiffEqFlux, Plots
using LaTeXStrings
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan")

function Schwarzschild_Geodesics(du, u, p, t)
    coord_time, r, θ, ϕ, p_t, p_r, p_θ, p_ϕ = u # state (Schwarzschild Coordinates)
    M, E, L = p # parameters (Mass, Energy, Angular Momentum)
    
    du[1] = dt = E*(1-2*M/r)^(-1)
    du[2] = dr = (1-2*M/r)*p_r
    # du[3] = dθ = 0
    du[4] = dϕ = r^(-2)*L

    # du[5] = dp_t = 0
    du[6] = dp_r = -(1/2)*( (1-2*M/r)^(-2)*(2*M/r^2)*(E)^2
                    + (2*M/(r^2))*(p_r)^2
                    - 2*(r^(-3))*(L)^2)
    # du[7] = dp_θ = 0
    # du[8] = dp_ϕ = 0
end

function Circular_Orbit_Energy(orbital_radius, BH_Mass)
    r = orbital_radius
    M = BH_Mass
    
    Energy_Squared = (1-(2*M/r))^2 / (1-(3*M/r))
    return sqrt(Energy_Squared)
end 

function Elliptical_Orbit_Energy(orbital_radius, BH_Mass, Momentum_r, Momentum_ϕ)
  r = orbital_radius
  M = BH_Mass
  H = -0.5
  Energy_Squared = -1*(1-(2*M/r)) * (2*H - r^(-2) * Momentum_ϕ^2 - (1-(2*M/r)) * Momentum_r^2)
  return sqrt(Energy_Squared)
end

function Angular_Momentum(orbital_radius, BH_Mass)
  r = orbital_radius
  M = BH_Mass
  Angular_Momentum_Squared = (M*r)/(1-(3*M)/r)
  return sqrt(Angular_Momentum_Squared)
end

M = 1.0 # Mass of Large Black Hole
R = 10*M # Radius of Orbit
BH_Kick = 0.2 # Initial Radial Momentum of Small Black Hole
E = Elliptical_Orbit_Energy(R, M, BH_Kick, Angular_Momentum(R, M)) # Energy of Orbit
L = Angular_Momentum(R, M) # Angular Momentum of Orbit

u0 = [0, R, 1.5707963267948966, 0, -1*E, BH_Kick, 0, L]
tspan = (0.0, 2000.0)
t = 0:0.1:2000.0
p = [M, E, L]

prob = ODEProblem(Schwarzschild_Geodesics, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = 0.1)

function predict_rd(p)
    prob_new = remake(prob, p = p)
    solve(prob_new, Tsit5(), saveat = 0.1)[1, :]
end

# sxs_h_plus = zeros(length(t))   # Dummy target data
# sxs_h_cross = zeros(length(t))  # Dummy target data

# r_orbit = 10.0
# M_bh = 1.0
# Ω_orbital = sqrt(M_bh/r_orbit^3)  # ≈ 0.0316
# f_gw = 2 * Ω_orbital              # ≈ 0.0632

# # Waveform amplitude (depends on orbiting mass and observer distance)
# amplitude = 0.001  # Adjust based on your system

# # Target waveform for circular orbit
# sxs_h_plus = amplitude * cos.(f_gw * t)
# sxs_h_cross = amplitude * sin.(f_gw * t)

function loss_waveform(p)
    sol = solve(remake(prob, p = p), Tsit5(), saveat = 0.1)

    predicted_waveform = compute_waveform(0.1, sol, 1.0)

    return sum(abs2, predicted_waveform[1] - sxs_h_plus) + 
            sum(abs2, predicted_waveform[2] - sxs_h_cross)
end

opt_state = Flux.setup(Flux.Adam(0.1), p)

cb = function()
    println("Loss: ", loss_waveform(p))
    
    # Solve with current parameters
    current_sol = solve(remake(prob, p = p), Tsit5(), saveat = 0.1)
    
    # Extract solution components
    y1 = current_sol[1,:] # coordinate time solutions
    y2 = current_sol[2,:] # radial coordinate solutions  
    y3 = current_sol[3,:] # equatorial angle solutions
    y4 = current_sol[4,:] # planar angle solutions
    
    # Create 2x2 subplot for all components
    p1 = plot(
        t, [y1 y2 y3 y4],
        layout = (2, 2),
        title = ["Coordinate Time" "Radius" "Equatorial Angle" "Angular Displacement"],
        xlabel = "Simulation Time",
        ylabel = ["t" "r" "θ" "ϕ"],
        label = ["Time" "Radius" "Theta" "Phi"],
        ylims = [(0, maximum(t)) (minimum(y2)-1, maximum(y2)+1) (1.35, 1.8) (0, maximum(y4))],
        linewidth = 2)
    
    # Create orbital trajectory plot
    particle_x = y2 .* cos.(y4)
    particle_y = y2 .* sin.(y4)
    
    p2 = plot(particle_x, particle_y,
        aspect_ratio = 1, linewidth = 2,
        title = "Schwarzschild Geodesic Trajectory",
        xlabel = "x",
        ylabel = "y",
        label = "Particle Trajectory")
    
    # Display both plots
    display(p1)
    display(p2)
end

# Initial callback
cb()

# Manual training loop with continuous visualization
for i in 1:100
    grads = Flux.gradient(loss_waveform, p)[1]
    Flux.update!(opt_state, p, grads)
    
    # Show progress every 10 iterations
    if i % 10 == 0
        println("Iteration $i")
        cb()
    end
end