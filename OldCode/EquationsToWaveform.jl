using Plots
using DifferentialEquations
using LaTeXStrings
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan")


function Schwarzschild_Geodesics(du, u, p, t)
    coord_time, r, θ, ϕ, p_t, p_r, p_θ, p_ϕ = u # state (Schwarzschild Coordinates)
    M, E, L = p # parameters (Mass, Energy, Angular Momentum)
    
    du[1] = dt = E*(1-2*M/r)^(-1)
    du[2] = dr = (1-2*M/r)*p_r
    du[3] = dθ = 0
    du[4] = dϕ = r^(-2)*L

    du[5] = dp_t = 0
    du[6] = dp_r = -(1/2)*( (1-2*M/r)^(-2)*(2*M/r^2)*(p_t)^2
                    + (2*M/(r^2))*(p_r)^2
                    - 2*(r^(-3))*(L)^2)
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
R = 10*M # Radius of Orbit
BH_Kick = 0.2 # Initial Radial Momentum of Small Black Hole
L = Angular_Momentum(R, M) # Angular Momentum of Orbit
E = Elliptical_Orbit_Energy(R, M, BH_Kick, L) # Energy of Orbit

u0 = [0, R, pi/2, 0, -1*E, BH_Kick, 0, L]
tspan = (0.0, 2000.0)
t = 0:0.1:2000.0
p_actual = [M, E, L]

prob = ODEProblem(Schwarzschild_Geodesics, u0, tspan, p_actual)
true_sol = solve(prob, Tsit5(), saveat = 0.1)

h_plus_training = compute_waveform(0.1, true_sol, 1.0)[1]
h_cross_training = compute_waveform(0.1, true_sol, 1.0)[2]

h_plus_training_EL = h_plus_training

plot(t, h_plus_training, label = "h+", linewidth = 3)

# function getParameters(solution)
#   r_max = maximum(solution[2,:])
#   r_min = minimum(solution[2,:])
#   e_true = (r_max-r_min)/(r_max+r_min) # True Eccentricity
#   p_true = (2*r_max*r_min)/(r_max+r_min) # True semi-latus rectum
#   return p_true, e_true
# end 

# getParameters(true_sol)