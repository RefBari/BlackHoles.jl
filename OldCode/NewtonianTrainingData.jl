using Plots
using DifferentialEquations
using LaTeXStrings
using Statistics
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan")

function pe_2_EL(semilatusRectum, eccentricity)
  p = semilatusRectum
  e = eccentricity
  
  M = 1
  E = (e^2 - 1)/(2*p)
  L = sqrt(p)

  return [M, E, L]
end

function Newtonian_Training(initial_conditions)
  semirectum = initial_conditions[1] # Semi-latus rectum
  ecc = initial_conditions[2] # Eccentricity

  function NewtonianHamiltonian(du, u, p, t)
    x = u # u[1] = t, u[2] = r, u[3] = θ, u[4] = ϕ  
    M, E, L = p
    
    function NewH(state_vec)
      t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec
      
      p = [p_t, p_r, p_θ, p_ϕ]
      g = [
            -(1 - 2/r)^(-1) 0 0 0;
            0 (1 + 2/r)^(-1) 0 0;
            0 0 0 0;
            0 0 0 (r^(-2))*(1 + 2/r)^(-1)
          ]

      H_schwarzschild = (1/2) * p' * g * p

      return H_schwarzschild # Returns equations of motion in PROPER time
    end
    
    # Compute gradient using ForwardDiff
    grad_H = ForwardDiff.gradient(NewH, x)

    # Define symplectic matrix J (8x8)
    J = [zeros(4,4)  I(4);
         -I(4)       zeros(4,4)]
    
    # Hamilton's equations: ẋ = J*∇H
    du_dτ = J * grad_H

    dH_dpₜ = grad_H[5]
    dτ_dt = (dH_dpₜ)^(-1)
    
    du .= du_dτ .* dτ_dt # du / dt Returns equations of motion in COORDINATE time
    du[1] = 1
  end

  M, E, L = pe_2_EL(semirectum, ecc)
  R = semirectum*M/(1+ecc)

  # Use relativistic constraint H = -1/2
  g_tt_factor = 1 - 2/R
  g_rr_factor = 1 + 2/R
  angular_term = (L^2 / R^2) * (1/g_rr_factor)
  p_t_squared = g_tt_factor * (1 + angular_term)  # For H = -1/2
  p_t_correct = -sqrt(p_t_squared)

  u0 = [0, R, π/2, 0, p_t_correct, 0, 0, L]

  tspan = (0.0, 6e4)
  timestep = 100
  t_full = 0:timestep:6e4
  p_actual = [M, E, L]

  prob = ODEProblem(NewtonianHamiltonian, u0, tspan, p_actual)
  true_sol = solve(prob, RK4(), saveat = timestep)

  y1 = true_sol[1,:] 
  y2 = true_sol[2,:] 
  y3 = true_sol[3,:] 
  y4 = true_sol[4,:] 

  # Create orbital trajectory plot
  particle_x = y2 .* cos.(y4)
  particle_y = y2 .* sin.(y4)

  p2 = plot(particle_x, particle_y,
                aspect_ratio = 1, linewidth = 2,
                title = "\n\nPredicted Trajectory",
                xlabel = L"x",
                ylabel = L"y",
                label = "Predicted Trajectory",
                bottom_margin = 10mm, top_margin = 10mm, left_margin = 10mm, right_margin = 10mm,
                color = "red")
        
  display(p2)

  h_plus_full = compute_waveform(timestep, true_sol, 1.0; coorbital=false)[1]
  h_cross_full = compute_waveform(timestep, true_sol, 1.0; coorbital=false)[2]

  skip_initial_points = 1
  skip_end_points = 1

  h_plus_training = h_plus_full[(skip_initial_points+1):end-skip_end_points]
  h_cross_training = h_cross_full[(skip_initial_points+1):end-skip_end_points]
  t_truncated = t_full[(skip_initial_points+1):end-skip_end_points]

  p = plot(t_truncated, h_plus_training, label=L"GW using $\dot{r},\dot{\phi}$ (Schwarzschild Coordinates)", 
        linewidth = 2, alpha = 0.5, color = "green", xlabel = "t", ylabel = "h₊",
        top_margin = 10mm, bottom_margin = 10mm, left_margin = 10mm, right_margin = 10mm)
  display(p)
  return [prob, true_sol, h_plus_training, h_cross_training]
  
  end

results = Newtonian_Training([100, 0.5])