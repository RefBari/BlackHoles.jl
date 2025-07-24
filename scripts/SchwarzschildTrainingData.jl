using Plots
using DifferentialEquations
using LaTeXStrings
using Statistics
include("/Users/rbari/Work/BlackHoles/scripts/WaveformMan.jl")

function pe_2_EL(semilatusRectum, eccentricity)
    p = semilatusRectum
    e = eccentricity

    M = 1
    E = sqrt((p-2-2*e)*(p-2+2*e) / (p*(p-3-e^2)))
    L = sqrt((p^2 * M^2) / (p-3-e^2))

    return [M, E, L]
end

function getParameters(solution, skipDataPoints = 0)
    solution_matrix = hcat([u for u in solution.u]...)

    skip_data = skipDataPoints

    # Remove first 100 columns 
    solution_matrix = solution_matrix[:, (skip_data+1):end]

    # Remove last 100 columns  
    solution_matrix = solution_matrix[:, 1:(end-skip_data)]

    r_max = maximum(solution_matrix[2, :])
    r_min = minimum(solution_matrix[2, :])
    e_true = (r_max-r_min)/(r_max+r_min) # True Eccentricity
    p_true = (2*r_max*r_min)/(r_max+r_min) # True semi-latus rectum

    return p_true, e_true
end

function create_Schwarzschild_trainingData(initial_conditions)
    semirectum = initial_conditions[1] # Semi-latus rectum
    ecc = initial_conditions[2] # Eccentricity

    function SchwarzschildHamiltonian_GENERIC(du, u, p, t)
        x = u # u[1] = t, u[2] = r, u[3] = θ, u[4] = ϕ  
        M, E, L = p

        function H(state_vec)
            t, r, θ, φ, p_t, p_r, p_θ, p_ϕ = state_vec

            f = (1 - ((2*M)/r))
            # print("\n\nTrue g_tt:", -f^(-1))
            p = [p_t, p_r, p_θ, p_ϕ]
            g = [
                -f^(-1) 0 0 0;
                0 f 0 0;
                0 0 0 0;
                0 0 0 r^(-2)
            ]

            H_schwarzschild = (1/2) * p' * g * p

            return H_schwarzschild # Returns equations of motion in PROPER time
        end

        # Compute gradient using ForwardDiff
        grad_H = ForwardDiff.gradient(H, x)

        # Define symplectic matrix J (8x8)
        J = [
            zeros(4, 4) I(4);
            -I(4) zeros(4, 4)
        ]

        # Hamilton's equations: ẋ = J*∇H
        du_dτ = J * grad_H

        t_val, r_val = x[1], x[2]
        f_val = 1 - 2*M/r_val
        dτ_dt = f_val/E

        du .= du_dτ .* dτ_dt # du / dt Returns equations of motion in COORDINATE time
    end

    M, E, L = pe_2_EL(semirectum, ecc)
    R = semirectum*M/(1+ecc) # Radius of Orbit

    u0 = [0, R, pi/2, 0, -1*E, 0, 0, L]
    tspan = (0.0, 20e4)
    timestep = 100
    t_full = 0:timestep:20e4
    p_actual = [M, E, L]

    prob = ODEProblem(SchwarzschildHamiltonian_GENERIC, u0, tspan, p_actual)
    true_sol = solve(prob, RK4(), saveat = timestep)

    y1 = true_sol[1, :]
    y2 = true_sol[2, :]
    y3 = true_sol[3, :]
    y4 = true_sol[4, :]

    # Create orbital trajectory plot
    particle_x = y2 .* cos.(y4)
    particle_y = y2 .* sin.(y4)

    p2 = plot(
        particle_x,
        particle_y,
        aspect_ratio = 1,
        linewidth = 2,
        title = "\n\nPredicted Trajectory",
        xlabel = L"x",
        ylabel = L"y",
        label = "Predicted Trajectory",
        bottom_margin = 10mm,
        top_margin = 10mm,
        left_margin = 10mm,
        right_margin = 10mm,
    )

    display(p2)

    h_plus_full = compute_waveform(timestep, true_sol, 1.0; coorbital = false)[1]
    h_cross_full = compute_waveform(timestep, true_sol, 1.0; coorbital = false)[2]

    skip_initial_points = 1
    skip_end_points = 1

    h_plus_training = h_plus_full[(skip_initial_points+1):(end-skip_end_points)]
    h_cross_training = h_cross_full[(skip_initial_points+1):(end-skip_end_points)]
    t_truncated = t_full[(skip_initial_points+1):(end-skip_end_points)]

    p = plot(
        t_truncated,
        h_plus_training,
        label = L"GW using $\dot{r},\dot{\phi}$ (Schwarzschild Coordinates)",
        linewidth = 2,
        alpha = 0.5,
        color = "green",
        xlabel = "t",
        ylabel = "h₊",
        top_margin = 10mm,
        bottom_margin = 10mm,
        left_margin = 10mm,
        right_margin = 10mm,
    )
    display(p)
    return [prob, true_sol, h_plus_training, h_cross_training]

end

# results = create_Schwarzschild_trainingData([10, 0.5])
# getParameters(solution, 0)
