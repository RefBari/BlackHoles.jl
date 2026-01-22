using Plots
using DifferentialEquations
using LaTeXStrings

function Schwarzschild_Geodesics(du, u, p, t)
    coord_time, r, θ, ϕ, p_t, p_r, p_θ, p_ϕ = u # state (Schwarzschild Coordinates)
    M, E, L = p # parameters (Mass, Energy, Angular Momentum)
    
    du[1] = dt = E*(1-2*M/r)^(-1)
    du[2] = dr = (1-2*M/r)*p_r
    du[3] = dθ = r^(-2)*p_θ
    du[4] = dϕ = r^(-2)*p_ϕ

    du[5] = dp_t = 0
    du[6] = dp_r = -(1/2)*( (1-2*M/r)^(-2)*(2*M/r^2)*(p_t)^2
                    + (2*M/(r^2))*(p_r)^2
                    - 2*(r^(-3))*(p_ϕ)^2)
    du[7] = dp_θ = 0
    du[8] = dp_ϕ = 0
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
R = 13*M # Radius of Orbit
BH_Kick = 0.24 # Initial Radial Momentum of Small Black Hole
L = Angular_Momentum(R, M) # Angular Momentum of Orbit
E = Elliptical_Orbit_Energy(R, M, BH_Kick, L) # Energy of Orbit

u0 = [0, R, pi/2, 0, -1*E, BH_Kick, 0, L]
tspan = (0.0, 6000.0)
t = 0:0.1:6000.0
p = [M, E, L]

prob = ODEProblem(Schwarzschild_Geodesics, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = 0.1)

y1 = sol[1,:] # coordinate time solutions
y2 = sol[2,:] # radial coordinate solutions
y3 = sol[3,:] # equatorial angle solutions
y4 = sol[4,:] # planar angle solutions

plot(
    t, [y1 y2 y3 y4],
    layout = (2, 2),
    title = ["Coordinate Time" "Radius" "Equatorial Angle" "Angular Displacement"],
    xlabel = "Simulation Time",
    ylabel = ["t" "r" "θ" "ϕ"],
    label = "Time",
    ylims = [(0, maximum(t)) (minimum(y2)-1, maximum(y2)+1) (1.35, 1.8) (0, maximum(y4))],
    linewidth = 3)

particle_x = y2 .* cos.(y4)
particle_y = y2 .* sin.(y4)

plot(particle_x, particle_y,
    aspect_ratio = 1, linewidth = 3,
    title = "Schwarzschild Geodesics",
    label = "Particle Trajectory")

anim = @animate for i in 1:50:length(t)
    plot(particle_x[1:i], particle_y[1:i],
        aspect_ratio = 1, 
        linewidth = 2, 
        color = :blue,
        label = "Trajectory",
        title = "Schwarzschild Geodesic (t = $(round(t[i], digits=1)))",
        xlabel = L"x",
        ylabel = L"y",
        xlims = (minimum(particle_x)-2, maximum(particle_x)+2),
        ylims = (minimum(particle_y)-2, maximum(particle_y)+2))
    
    scatter!([particle_x[i]], [particle_y[i]],
                color = :red, markersize = 8,
                label = "Particle")

    scatter!([0], [0], color = :black, markersize = 10, label = "Black Hole")

end

mp4(anim, "crazy_schwarzschild_orbit.mp4", fps = 15)