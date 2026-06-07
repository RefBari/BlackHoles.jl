```
    Nested Inverse Problem: Gravitational Wave to Orbits to Metric
    Learns Only Schwarzschild Metric + Orbits + Waveform for Conservative Dynamics!

```

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using CairoMakie
using Statistics
using JLD2
using PrettyTables
using Printf
using PrettyTables: ft_printf

include("TrainingDataFunctions.jl")
include("PreviousNNFunctions.jl")
include("Orbit2Waveform.jl")
include("DissipationRicci.jl")

set_theme!(
    fontsize   = 16,
    fonts      = (; regular = "Latin Modern Roman",   # LaTeX default font
                    bold    = "Latin Modern Roman",
                    italic  = "Latin Modern Roman"),
    Axis = (
        titlesize      = 18,
        xlabelsize     = 16,
        ylabelsize     = 16,
        xticklabelsize = 13,
        yticklabelsize = 13,
        titlefont      = :bold,
        spinewidth     = 1.5,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xtickalign     = 1,    # ticks point inward
        ytickalign     = 1,
    ),
    Legend = (
        framevisible   = true,
        framewidth     = 1.0,
        labelsize      = 14,
    ),
    Lines = (
        linewidth = 2.0,
    ),
)

""" ************************
STEP 1: DEFINE INITIAL CONDITIONS
************************ """
tspan = (0, 1e3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize) 

rp_true = 2.5
ra_true = 8
θmin_true = π/4
a_true = 0.95
mass_ratio = 1e3

r_min, r_max = rp_true, ra_true

noise = 0

""" ************************
STEP 2: IMPORT TRAINING DATA
************************ """

# Gather waveform data

waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Kerr_rp10_ra20noise0.0_massRatio1000.0spin_0.9inclination_0.7853981633974483_time6000.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Kerr_rp10_ra20noise0.0_massRatio1000.0spin_0.9inclination_0.7853981633974483_time6000.txt")
true_solution = file2solution(tsteps,"input/solution_Kerr_rp10_ra20noise0.0_massRatio1000.0spin_0.9inclination_0.7853981633974483_time6000.txt")

""" *******************************
STEP 3: DEFINE SIMULATION PARAMETERS
******************************* """
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = mass_ratio/(1.0+mass_ratio)
mass2 = 1.0/(1.0+mass_ratio)

dt_data = tsteps[2] - tsteps[1]
dt = 1
num_optimization_increments = 10

""" *******************************
STEP 4: CREATE NEURAL NETWORKS
******************************* """

NN_Conservative = Chain(
    Dense(2, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 5),
)

NN_Dissipative = Chain(
    Dense(1, 10, tanh), # Input: r
    Dense(10, 10, tanh),
    Dense(10, 1),
)

""" *******************************
STEP 5: INITIALIZE NEURAL NETWORKS
******************************* """
# Initialize parameters for both NNs
rng = MersenneTwister(222)
NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)
NN_Dissipative_params, NN_Dissipative_state = Lux.setup(rng, NN_Dissipative)

NN_Conservative_params = Lux.fmap(x -> Float64.(x), NN_Conservative_params)
NN_Dissipative_params  = Lux.fmap(x -> Float64.(x), NN_Dissipative_params)

initialize_Conservative_NN(1e-3, [-0.1, -0.1, -0.1, -0.1, -0.1], 0.3, 0.0)
initialize_Dissipative_NN(1e-2, -4, -1, -1)

""" ******************************************************************
STEP 6: ASSIGN NEURAL NETWORK INPUTS & EXTRACT NEURAL NETWORK OUTPUTS
****************************************************************** """
NN_params = ComponentArray(
    conservative = NN_Conservative_params, 
    dissipative = NN_Dissipative_params
)

""" ********************************************************
STEP 6A: CREATE HELPER FUNCTION TO CONSTRUCT INITIAL CONDITION
******************************************************** """

function solve_ELQ(rₚ, rₐ, θmin; M = 1.0, a = a_true, μ = 1.0)
    Δₚ = rₚ^2 - 2*M*rₚ + a^2
    Δₐ = rₐ^2 - 2*M*rₐ + a^2

    function equations!(F, x)
        E, Lz = x[1], x[2]
        Q_from_rₚ = (((rₚ^2 + a^2) * E - a * Lz)^2 - Δₚ * ((Lz-a*E)^2 + μ^2 * rₚ^2) ) / Δₚ

        F[1] = ((rₐ^2 + a^2) * E - a * Lz)^2 - Δₐ * ( (Lz - a*E)^2 + μ^2 * rₐ^2 + Q_from_rₚ )
        F[2] = Q_from_rₚ - cos(θmin)^2 * (a^2 * (μ^2 - E^2) + Lz^2 / sin(θmin)^2 )
    end

    # Initial Guesses
    E0 = sqrt( 1 - 2 * M / (rₚ + rₐ) )
    L0 = sqrt( M * (rₚ + rₐ) / 2 )
    sol = nlsolve(equations!, [E0, L0]; autodiff=:forward, ftol = 1e-12)

    E, Lz = sol.zero
    Δₚ_val = rₚ^2 - 2*M*rₚ + a^2
    Q = ( ((rₚ^2 + a^2) * E - a * Lz)^2 - Δₚ_val *  ( (Lz - a*E)^2 + μ^2 * rₚ^2 ) ) / Δₚ_val

    return E, Lz, Q
end

function make_u0(rₚ, rₐ, θmin; M = 1.0, a = a_true, μ = 1.0)
    E, Lz, Q = solve_ELQ(rₚ, rₐ, θmin; M = M, a = a, μ = μ)
    return [0.0, rₚ, θmin, 0.0, 
            -E, 0.0, 0.0, Lz]
end

""" ********************************
STEP 7: CREATE FUNCTION FOR ODE MODEL
*********************************"""

function ODE_model(du, u, p, t)
    du = Newtonian(ra_true, du, u, model_params, t; NN = NN_adapter, NN_params = p)
    return du
end

""" ********************************************************
STEP 8: DEFINE & SOLVE ODE MODEL + CONVERT ORBIT TO WAVEFORM
*********************************************************"""

u0_init = make_u0(rp_true, ra_true, θmin_true)
prob_nn = ODEProblem(ODE_model, u0_init, tspan, NN_params)
soln_nn = Array(solve(prob_nn, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))

waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_nn, mass_ratio, a_true; coorbital=false)
orbit = soln2orbit(soln_nn; a = a_true)
pred_orbit1_init, pred_orbit2_init = one2two(orbit, mass1, mass2)

orbit_plot, h₊_plot, hₓ_plot = plot_orbits_waveforms()

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """

bias_history = []

rm("training_log.txt", force=true)

function plot_geodesic_3d!(gp, pred_soln, true_soln; M = 1.0, a = a_true)
    N = size(pred_soln, 2)

    r, θ, ϕ       = pred_soln[2,:], pred_soln[3,:], pred_soln[4,:]
    r_t, θ_t, ϕ_t = true_soln[2,1:N], true_soln[3,1:N], true_soln[4,1:N]

    x   = @. sqrt(r^2+a^2)   * sin(θ)   * cos(ϕ);   y   = @. sqrt(r^2+a^2)   * sin(θ)   * sin(ϕ);   z   = @. r   * cos(θ)
    x_t = @. sqrt(r_t^2+a^2) * sin(θ_t) * cos(ϕ_t); y_t = @. sqrt(r_t^2+a^2) * sin(θ_t) * sin(ϕ_t); z_t = @. r_t * cos(θ_t)

    ax = CairoMakie.Axis3(gp, aspect = :data,
                          xlabel = L"x", ylabel = L"y", zlabel = L"z",
                          title  = "Geodesic")

    CairoMakie.lines!(ax, x_t, y_t, z_t; linewidth = 2, label = "True")
    CairoMakie.lines!(ax, x,   y,   z;   linewidth = 2, label = "Predicted", linestyle = :dash)

    rH = M + sqrt(M^2 - a^2)
    θg = range(0, π,  length = 25)
    ϕg = range(0, 2π, length = 50)
    X = [rH*sin(th)*cos(ph) for th in θg, ph in ϕg]
    Y = [rH*sin(th)*sin(ph) for th in θg, ph in ϕg]
    Z = [rH*cos(th)         for th in θg, ph in ϕg]
    CairoMakie.surface!(ax, X, Y, Z; alpha = 0.15)

    return ax
end

function plot_final_biases!(gp, p)
    gl = CairoMakie.GridLayout(gp)
    
    biases = p.conservative.layer_3.bias
    n_outputs = length(biases)
    
    # Reshape to column vector for heatmap
    bias_matrix = reshape(biases, (n_outputs, 1))
    
    ax = CairoMakie.Axis(gl[1, 1];
        title = "Final Layer Biases",
        xlabel = "",
        ylabel = "Output Index",
        yticks = (1:n_outputs, ["Out $i" for i in 1:n_outputs]),
        xticks = ([0.5], ["Bias"])
    )
    
    # Heatmap
    max_abs = maximum(abs.(biases))
    hm = CairoMakie.heatmap!(ax, [0.5], 1:n_outputs, bias_matrix;
        colormap = :RdBu,
        colorrange = (-max_abs, max_abs))
    
    # Add text labels with actual values
    for i in 1:n_outputs
        CairoMakie.text!(ax, 0.5, i;
            text = @sprintf("%.3e", biases[i]),
            align = (:center, :center),
            fontsize = 14,
            color = abs(biases[i]) > 0.5*max_abs ? :white : :black)
    end
    
    Colorbar(gl[2, 1], hm;
             label = "Bias Value",
             vertical = false,
             flipaxis = false,
             tellwidth = false)
    
    return ax
end

function loss(NN_params; saveat = tsteps, ricci_weight = 0.0)
    function metric(x)
        r_val = x[2]
        θ_val = x[3]
        output, _ = NN_Conservative([r_val / r_max, (cos(θ_val))^2],
                                    NN_params.conservative, NN_Conservative_state)
        return g_NeuralNetwork(x, output)
    end

    # METRIC LOSS (no ODE loss)    
    r_check = range(2.5, r_max, length = 32)
    θ_check = range(π/15, 14π/15, length = 32)
    
    ricci_loss = 0.0

    for radius in r_check
        for angle in θ_check
            input_state = @SVector [0, radius, angle, 0.0]
            Riemann_Tensor = Riemann(metric, input_state)
            Ricci_covariant = RicciTensor(Riemann_Tensor)
            ricci_loss += sum(abs2, Ricci_covariant)
        end
    end
    
    ricci_loss = ricci_loss / (length(r_check) * length(θ_check))

    tspan = (saveat[1], saveat[end])
    u0_local = make_u0(rp_true, ra_true, θmin_true)
    prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = solve(prob_pred, Tsit5(); saveat = saveat, dt = dt, adaptive = false, verbose = false)

    N_expected = length(saveat)
    L2_metric_loss = 0
    if size(pred_soln, 2) < N_expected || any(isnan, pred_soln) || any(isinf, pred_soln)
        dummy_wave = zeros(N_expected)
        val_wave = 1e10
        return (L2_metric_loss, val_wave, 0.0, 0.0, 0.0, ForwardDiff.value(L2_metric_loss)),
                dummy_wave, dummy_wave, pred_soln
    end

    pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
    N = length(pred_waveform_real)
    val_wave = sum(abs2, waveform_real_ecc[1:N] .- pred_waveform_real) / sum(abs2, waveform_real_ecc[1:N]) +
                sum(abs2, waveform_imag_ecc[1:N] .- pred_waveform_imag) / sum(abs2, waveform_imag_ecc[1:N])
    val_ricci = 1e5 * ricci_loss

    loss = val_wave + val_ricci

    return (loss, ForwardDiff.value(val_wave), ForwardDiff.value(val_ricci), 0.0, 0.0, ForwardDiff.value(L2_metric_loss)),
            pred_waveform_real, pred_waveform_imag, pred_soln
end

function plot_loss!(gp, losses, waveform_loss, ricci_losses, flatn_loss, bound_loss, metric_loss)
    ax = CairoMakie.Axis(gp,
                        title  = L"Training Loss $\mathcal{L}$",
                        xlabel = "Iteration",
                        ylabel = L"\mathcal{L}",
                        yscale = log10)
    iter = 1:length(losses)
    CairoMakie.lines!(ax, collect(iter), losses;        linewidth = 2, label = "Total", linestyle = :solid)
    CairoMakie.lines!(ax, collect(iter), waveform_loss; linewidth = 2, label = "Waveform", linestyle = :dash)
    CairoMakie.lines!(ax, collect(iter), ricci_losses;  linewidth = 2, label = "Ricci", linestyle = :dash)
    # CairoMakie.lines!(ax, collect(iter), metric_loss; linewidth = 2, label = "Metric", linestyle = :solid)
    CairoMakie.axislegend(ax, position = :rt)
    return ax
end

# Test loss function
loss(NN_params)

losses = []
waveform_loss = []
ricci_losses = []
flatness_losses = []
boundOrbit_losses = []
metric_losses = []

L2_norms_gtt = []
L2_norms_grr = []
L2_norms_dissipation = []

max_total_iterations = 1000

""" *************************************
STEP 10: RUN BFGS OPTIMIZATION ALGORITHM
************************************ """

optimization_increments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
progress_bar = Progress(length(optimization_increments);
                        dt = 1.0, color=:green, desc = "Optimization Progress", 
                        barlen = 40)

function compute_rel_L2(p, r_range, θ_range)
    components = [(1,1), (2,2), (3,3), (4,4), (1,4)]
    errors = zeros(5)
    
    for (idx, (i,j)) in enumerate(components)
        num = 0.0
        den = 0.0
        for r in r_range, θ in θ_range
            sv = @SVector [0.0, r, θ, 0.0]
            out, _ = NN_Conservative([r / r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
            nn_val = g_NeuralNetwork(sv, out)[i,j]
            true_val = g_Kerr(sv; M = 1.0, a = a_true)[i,j]
            num += (nn_val - true_val)^2
            den += true_val^2
        end
        errors[idx] = den > 0 ? sqrt(num / den) : sqrt(num)
    end
    
    return errors  # [tt, rr, θθ, ϕϕ, tϕ]
end

best_loss = Inf
best_NN_params = deepcopy(NN_params)

for i in optimization_increments
    println("\noptimization increment :: ", i, " of ", num_optimization_increments)
    opt_first = optimization_increments[1]         # 40
    opt_last  = optimization_increments[end]       # 100
    frac = 0.10 + 0.95 * (i - opt_first) / (opt_last - opt_first)

    t_end = tspan[1] + frac * (tspan[2] - tspan[1])
    tsteps_increment = tsteps[tsteps .<= t_end]

    open("training_log.txt", "a") do io  # Changed "w" to "a" to append, not overwrite
        println(io, "\n===============================================")
        println(io, "Training increment ", i, "/", num_optimization_increments,
                ": using ", length(tsteps_increment), " of ", length(tsteps),
                " points (", round(frac*100; digits=1), "% of total data)")
        println(io, "===============================================")
    end

    tmp_loss(p) = loss(p, saveat = tsteps_increment, ricci_weight = current_ricci_weight)
    function scalar_loss(p)
        (loss_val, _, _, _, _, _), _, _, _  = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function build_dashboard(p, pred_soln, pred_waveform_real,
                         tsteps, N, startPoint, losses, waveform_loss,
                         ricci_losses, flatness_loss, bound_loss, metric_loss, loss_val)

        fig = CairoMakie.Figure(size = (1800, 1600))

        # ── Row 1: diagnostics ──────────────────────────────────────────
        plot_waveform!(fig[1, 1], tsteps, waveform_real_ecc,
                    pred_waveform_real, startPoint, N)
        plot_geodesic_3d!(fig[1, 2], pred_soln, true_solution)
        plot_loss!(fig[1, 3], losses, waveform_loss, ricci_losses, flatness_loss, bound_loss, metric_loss)

        # ── Rows 2–3: metric surfaces ───────────────────────────────────
        plot_metric_components!(fig[2:3, :], p, r_min, r_max, a_true)   # pass row range

        # plot_gtt_relative_error!(fig[3, 3], p, r_min, r_max)
        # plot_final_biases!(fig[3, 3], p)
        plot_bias_evolution!(fig[3, 3], bias_history)
        # ── Super title ─────────────────────────────────────────────────
        title_str = latexstring(
            "\\text{Iteration\\; $(length(losses))} \\;|\\; " *
            "a=$(a_true),\\; r_{\\min}=$(r_min),\\; r_{\\max}=$(r_max),\\; \\theta_{\\min}=$(round(θmin_true, sigdigits=4)) \\;|\\; " *
            "\\mathcal{L} = $(round(loss_val, sigdigits=4))"
        )

        Label(fig[0, :], title_str;
      fontsize = 30, font = :bold, padding = (0, 0, 12, 0))

        colgap!(fig.layout, 25)
        rowgap!(fig.layout, 25)

        return fig
    end

    function plot_bias_evolution!(gp, bias_history)
        gl = CairoMakie.GridLayout(gp)
        
        n_iters = length(bias_history)
        n_outputs = length(bias_history[1])
        
        # Stack into matrix: rows = iterations, cols = outputs
        bias_matrix = stack(bias_history; dims=1)
        
        ax = CairoMakie.Axis(gl[1, 1];
            title = "Final Layer Bias Evolution",
            xlabel = "Iteration",
            ylabel = "Bias Value"
        )
        
        # Get max absolute value across all biases for normalization
        max_abs = maximum(abs.(bias_matrix))
        
        metric_labels = [L"g_{tt}", L"g_{rr}", L"g_{\theta\theta}", L"g_{\phi\phi}", L"g_{t\phi}"]

        # Plot each output with intensity based on final value magnitude
        for i in 1:n_outputs
            final_value = bias_matrix[end, i]
            
            # Normalize to [0, 1] for opacity
            intensity = abs(final_value) / max_abs
            
            # Red for negative, blue for positive
            if final_value < 0
                line_color = RGBAf(1.0, 0.0, 0.0, intensity)  # Red with varying opacity
            else
                line_color = RGBAf(0.0, 0.0, 1.0, intensity)  # Blue with varying opacity
            end
            
            CairoMakie.lines!(ax, 1:n_iters, bias_matrix[:, i];
                linewidth = 2,
                color = line_color,
                label = metric_labels[i])    
        end
        
        CairoMakie.axislegend(ax, position = :rt)
        
        return ax
    end
    
    function opt_callback(state, args...; kwargs...)
        p = state.u

        (loss_val, v_wave, v_ricci, v_flat, v_bound, v_metric), pred_waveform_real, _, pred_soln  = tmp_loss(p)

        global NN_params, best_loss, best_NN_params

        current_loss = ForwardDiff.value(loss_val)

        NN_params = deepcopy(p)

        if current_loss < best_loss
            best_loss = current_loss
            best_NN_params = deepcopy(p)
            @save "best_NN_params.jld2" best_NN_params best_loss
        end

        push!(losses, max(loss_val, 1e-20))
        push!(waveform_loss, max(v_wave + 1e-20, 1e-20))
        push!(ricci_losses, max(v_ricci + 1e-20, 1e-20))
        push!(flatness_losses, max(v_flat + 1e-20, 1e-20))
        push!(boundOrbit_losses, max(v_bound + 1e-20, 1e-20))
        push!(metric_losses, max(v_metric + 1e-20, 1e-20))

        if length(losses) >= max_total_iterations
            print("Reached Maximum Total Iterations of ", max_total_iterations, ". Stopping Optimization.")
            return true
        end

        function metric(x)
            r_val = x[2]
            θ_val = x[3]
            scale_factor = r_max
            output, _ = NN_Conservative([r_val / scale_factor, (cos(θ_val))^2], p.conservative, NN_Conservative_state)
            return g_NeuralNetwork(x, output)
        end
        
        sample_vector = @SVector [0, 10.0, π/2, 0.0]
        b_track = ForwardDiff.value.(p.conservative.layer_3.bias)
        pred_metric = metric(sample_vector)
        true_metric = g_Kerr(sample_vector; M = 1.0, a = a_true)

        r_diag = range(r_min, r_max * 2, length = 20)
        θ_diag = [π/6, π/4, π/3, π/2]
        rel_L2 = compute_rel_L2(p, r_diag, θ_diag)
        rel_L2_tt, rel_L2_rr, rel_L2_θθ, rel_L2_ϕϕ, rel_L2_tϕ = rel_L2
        
        push!(bias_history, copy(p.conservative.layer_3.bias))

        open("training_log.txt", "a") do io
            println(io, "═"^50)
            println(io, "Iteration $(length(losses)) / $(Int(max_total_iterations))")
            println(io, "─"^50)
            
            # Loss summary
            @printf(io, "  Total Loss       : %.6e\n", loss_val)
            @printf(io, "  Waveform Loss    : %.6e\n", v_wave)
            @printf(io, "  Ricci Loss       : %.6e\n", v_ricci)
            @printf(io, "  Flatness Loss    : %.6e\n", v_flat)
            @printf(io, "  Bound Orbit Loss : %.6e\n", v_bound)
            
            println(io, "─"^50)
            println(io, "Componentwise metric diagnostics:")
            
            r_table = [8.0, 12.0, 16.0, 20.0]
            θ_table = [π/8, π/4, π/3, 5π/12, π/2]
            
            # ── Ricci table ──────────────────────────────────────────────
            println(io, "\n  Ricci ||R_μν||² by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %12.1f", r); end
            println(io)
            println(io, "  " * "─"^60)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    R = Riemann(metric, sv)
                    Ric = RicciTensor(R)
                    @printf(io, "  %12.3e", sum(abs2, Ric))
                end
                println(io)
            end
            
            # ── g_tt table: value (% error) ──────────────────────────────
            println(io, "\n  g_tt: predicted (% error) by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %16.1f", r); end
            println(io)
            println(io, "  " * "─"^76)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    out, _ = NN_Conservative([r/r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
                    nn_val   = g_NeuralNetwork(sv, out)[1,1]
                    true_val = g_Kerr(sv)[1,1]
                    pct      = 100*abs(nn_val - true_val)/abs(true_val)
                    @printf(io, "  %7.4f(%5.1f%%)", nn_val, pct)
                end
                println(io)
            end
            
            # ── g_rr table: value (% error) ──────────────────────────────
            println(io, "\n  g_rr: predicted (% error) by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %16.1f", r); end
            println(io)
            println(io, "  " * "─"^76)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    out, _ = NN_Conservative([r/r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
                    nn_val   = g_NeuralNetwork(sv, out)[2,2]
                    true_val = g_Kerr(sv)[2,2]
                    pct      = 100*abs(nn_val - true_val)/abs(true_val)
                    @printf(io, "  %7.4f(%5.1f%%)", nn_val, pct)
                end
                println(io)
            end

            # -- g_θθ table: value (% error) --─────────────────────────────
            println(io, "\n  g_θθ: predicted (% error) by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %16.1f", r); end
            println(io)
            println(io, "  " * "─"^76)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    out, _ = NN_Conservative([r/r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
                    nn_val   = g_NeuralNetwork(sv, out)[3,3]
                    true_val = g_Kerr(sv)[3,3]
                    pct      = 100*abs(nn_val - true_val)/abs(true_val)
                    @printf(io, "  %7.4f(%5.1f%%)", nn_val, pct)
                end
                println(io)
            end

            # -- g_ϕϕ table: value (% error) --─────────────────────────────
            println(io, "\n  g_ϕϕ: predicted (% error) by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %16.1f", r); end
            println(io)
            println(io, "  " * "─"^76)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    out, _ = NN_Conservative([r/r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
                    nn_val   = g_NeuralNetwork(sv, out)[4,4]
                    true_val = g_Kerr(sv)[4,4]
                    pct      = 100*abs(nn_val - true_val)/abs(true_val)
                    @printf(io, "  %7.4f(%5.1f%%)", nn_val, pct)
                end
                println(io)
            end

            # -- g_tϕ table: value (% error) --─────────────────────────────
            println(io, "\n  g_tϕ: predicted (% error) by (θ, r):")
            @printf(io, "  %-8s", "θ \\ r")
            for r in r_table; @printf(io, "  %16.1f", r); end
            println(io)
            println(io, "  " * "─"^76)
            for θ in θ_table
                @printf(io, "  %-8.3f", θ)
                for r in r_table
                    sv = @SVector [0.0, r, θ, 0.0]
                    out, _ = NN_Conservative([r/r_max, cos(θ)^2], p.conservative, NN_Conservative_state)
                    nn_val   = g_NeuralNetwork(sv, out)[1,4]
                    true_val = g_Kerr(sv)[1,4]
                    pct      = 100*abs(nn_val - true_val)/abs(true_val)
                    @printf(io, "  %7.4f(%5.1f%%)", nn_val, pct)
                end
                println(io)
            end
        end

        N = length(pred_waveform_real)
        startPoint = 1

        fig = build_dashboard(p, pred_soln, pred_waveform_real,
                            tsteps, N, startPoint,
                            losses, waveform_loss, ricci_losses, flatness_losses, boundOrbit_losses, metric_losses, loss_val)
        display(fig)

        return false
    end

    p_init = ComponentArray(NN_params)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> scalar_loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)

    if i == 1
        current_ricci_weight = 10
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try
            res = Optimization.solve(optprob,
                Optim.BFGS(; initial_stepnorm=5f0, linesearch=LineSearches.BackTracking()),
                callback=opt_callback, allow_f_increases=false, maxiters=100)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end        
    elseif i == 2
        current_ricci_weight = 5
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try
            res = Optimization.solve(optprob,
                Optim.BFGS(; initial_stepnorm=4f0, linesearch=LineSearches.BackTracking()),
                callback=opt_callback, allow_f_increases=false, maxiters=80)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    elseif i == 3
        current_ricci_weight = 4
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(; initial_stepnorm=3f0, linesearch = LineSearches.BackTracking()), 
                                    callback=opt_callback, 
                                    allow_f_increases=false, maxiters = 40)
            global NN_params = res.minimizer
        catch e 
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    elseif i == 4
        current_ricci_weight = 3
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(; initial_stepnorm=2f0, linesearch = LineSearches.BackTracking()), 
                                    callback=opt_callback, 
                                    allow_f_increases=false, maxiters = 20)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    elseif i == 5
        current_ricci_weight = 2
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(; initial_stepnorm=1f0, linesearch = LineSearches.BackTracking()), 
                                    callback=opt_callback, 
                                    allow_f_increases=false, maxiters = 10)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    elseif i == 6
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(; initial_stepnorm=0.1f0, linesearch = LineSearches.BackTracking()), 
                                    callback=opt_callback, 
                                    allow_f_increases=false, maxiters = 20)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    elseif i == 7
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                Optim.BFGS(; initial_stepnorm=0.1f0, linesearch = LineSearches.BackTracking()), 
                                callback=opt_callback, 
                                allow_f_increases=false, maxiters = 40)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    else
        current_ricci_weight = 1
        print("Ricci weight at iteration ", i, " is ", current_ricci_weight, "\n")
        try 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(; initial_stepnorm=0.1f0, linesearch = LineSearches.BackTracking()), 
                                    callback=opt_callback, 
                                    allow_f_increases=false, maxiters = 45)
            global NN_params = res.minimizer
        catch e
            println("Optimization failed at increment $i: ", e)
            println("Continuing with current parameters...")
        end
    end

    percent_data = round(frac*100; digits = 1)
    current_loss = length(losses) > 0 ? losses[end] : 0.0

    next!(progress_bar; showvalues = [
        (:Increment, "$(i) / $(optimization_increments[end])"),
        (:Data_Size, "$(percent_data)%"),
        (:Current_Loss, current_loss)
    ])
end

@load "best_NN_params.jld2" best_NN_params best_loss
NN_params = best_NN_params

fig = CairoMakie.Figure(size = (1800, 1600))
u0_local = make_u0(rp_true, ra_true, θmin_true)
prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
pred_soln = Array(solve(prob_pred, Tsit5(), saveat=tsteps, dt=dt, adaptive=false))
pred_waveform = compute_waveform(dt_data, pred_soln, mass_ratio)
true_waveform = compute_waveform(dt_data, true_solution, mass_ratio)
L2_waveform_final = norm(pred_waveform .- true_waveform)

fig = CairoMakie.Figure(size=(800, 600))
ax = CairoMakie.Axis(fig[1,1], xlabel="Time", ylabel="r", title="r vs Time (First 100s)")
CairoMakie.plot!(ax, tsteps, true_waveform; label="True", linewidth=2)
CairoMakie.plot(ax, tsteps, pred_waveform; label="Predicted", linewidth=2, linestyle=:dash)

plot_geodesic_3d!(fig[1,1], pred_soln, true_solution)
display(fig)

# first 100s of r 
r_predicted = pred_soln[2, 1:85]
r_true = true_solution[2, 1:85]

L2_r_norm = norm(r_predicted - r_true)
final_time = tsteps[85]
error_r = L2_r_norm * final_time

times = tsteps[1:85]
fig = CairoMakie.Figure(size=(800, 600))
ax = CairoMakie.Axis(fig[1,1], xlabel="Time", ylabel="r", title="r vs Time (First 100s)")
CairoMakie.lines!(ax, times, r_true; label="True", linewidth=2)
CairoMakie.lines!(ax, times, r_predicted; label="Predicted", linewidth=2, linestyle=:dash)
display(fig)

w1, b1, w2, b2, w3, b3 = get_weights_biases(NN_params)

# ── SETUP ────────────────────────────────────────────────────────────────────
mkpath("plots/final")

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
true_waveform_real, true_waveform_imag = compute_waveform(dt_data, true_solution, mass_ratio)
N = length(pred_waveform_real)

r_vals  = range(r_min, r_max, length = 60)
θ_vals  = range(1e-3, π/2,   length = 60)

function nn_metric(r, θ)
    sv    = @SVector [0.0, r, θ, 0.0]
    out, _ = NN_Conservative([r/r_max, cos(θ)^2],
                              NN_params.conservative, NN_Conservative_state)
    return g_NeuralNetwork(sv, out)
end

# ── 1. GRAVITATIONAL WAVE ─────────────────────────────────────────────────────
fig1 = CairoMakie.Figure(size = (900, 500))
ax   = CairoMakie.Axis(fig1[1, 1];
        title  = L"Gravitational Wave $h_+$",
        xlabel = L"t\ (M)", ylabel = L"h_+")
CairoMakie.lines!(ax, tsteps[1:N], true_waveform_real[1:N]; linewidth = 2, label = "True")
CairoMakie.lines!(ax, tsteps[1:N], pred_waveform_real[1:N]; linewidth = 2, linestyle = :dash, label = "Predicted")
CairoMakie.axislegend(ax)
CairoMakie.save("plots/final/1_waveform.png", fig1)
display(fig1)

# ── 2. 3D GEODESIC ───────────────────────────────────────────────────────────
fig2 = CairoMakie.Figure(size = (900, 800))
plot_geodesic_3d!(fig2[1, 1], pred_soln, true_solution)
CairoMakie.save("plots/final/2_geodesic_3d.png", fig2)
display(fig2)

# ── 3. PHASE PLOT ─────────────────────────────────────────────────────────────
fig3 = CairoMakie.Figure(size = (900, 500))
plot_phase!(fig3[1, 1], pred_soln, true_solution)
CairoMakie.save("plots/final/3_phase.png", fig3)
display(fig3)

# ── 4. g_tt SURFACE ──────────────────────────────────────────────────────────
g_tt_true = [g_Kerr(@SVector [0.0, r, θ, 0.0], M = 1.0, a = a_true)[1,1] for r in r_vals, θ in θ_vals]
g_tt_nn   = [nn_metric(r, θ)[1,1]                    for r in r_vals, θ in θ_vals]
clims_tt  = (min(minimum(g_tt_true), minimum(g_tt_nn)),
             max(maximum(g_tt_true), maximum(g_tt_nn)))

fig4 = CairoMakie.Figure(size = (1200, 500))
ax4a = CairoMakie.Axis3(fig4[1, 1]; title = L"True $g_{tt}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{tt}",
        azimuth = 1.3π, elevation = 0.2π)
ax4b = CairoMakie.Axis3(fig4[1, 2]; title = L"Predicted $g_{tt}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{tt}",
        azimuth = 1.3π, elevation = 0.2π)
CairoMakie.surface!(ax4a, collect(r_vals), collect(θ_vals), g_tt_true;
        colormap = :heat, colorrange = clims_tt)
CairoMakie.surface!(ax4b, collect(r_vals), collect(θ_vals), g_tt_nn;
        colormap = :viridis, colorrange = clims_tt)
CairoMakie.save("plots/final/4_gtt.png", fig4)
display(fig4)

# ── 5. g_rr SURFACE ──────────────────────────────────────────────────────────
g_rr_true = [g_Kerr(@SVector [0.0, r, θ, 0.0], M = 1.0, a = a_true)[2,2] for r in r_vals, θ in θ_vals]
g_rr_nn   = [nn_metric(r, θ)[2,2]                    for r in r_vals, θ in θ_vals]
clims_rr  = (min(minimum(g_rr_true), minimum(g_rr_nn)),
             max(maximum(g_rr_true), maximum(g_rr_nn)))

fig5 = CairoMakie.Figure(size = (1200, 500))
ax5a = CairoMakie.Axis3(fig5[1, 1]; title = L"True $g_{rr}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{rr}",
        azimuth = 1.3π, elevation = 0.2π)
ax5b = CairoMakie.Axis3(fig5[1, 2]; title = L"Predicted $g_{rr}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{rr}",
        azimuth = 1.3π, elevation = 0.2π)
CairoMakie.surface!(ax5a, collect(r_vals), collect(θ_vals), g_rr_true;
        colormap = :heat, colorrange = clims_rr)
CairoMakie.surface!(ax5b, collect(r_vals), collect(θ_vals), g_rr_nn;
        colormap = :viridis, colorrange = clims_rr)
CairoMakie.save("plots/final/5_grr.png", fig5)
display(fig5)

# ── 6. g_tϕ SURFACE ──────────────────────────────────────────────────────────
g_tp_true = [g_Kerr(@SVector [0.0, r, θ, 0.0], M = 1.0, a = a_true)[1,4] for r in r_vals, θ in θ_vals]
g_tp_nn   = [nn_metric(r, θ)[1,4]                    for r in r_vals, θ in θ_vals]
clims_tp  = (min(minimum(g_tp_true), minimum(g_tp_nn)),
             max(maximum(g_tp_true), maximum(g_tp_nn)))

fig6 = CairoMakie.Figure(size = (1200, 500))
ax6a = CairoMakie.Axis3(fig6[1, 1]; title = L"True $g_{t\phi}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{t\phi}",
        azimuth = 1.3π, elevation = 0.2π)
ax6b = CairoMakie.Axis3(fig6[1, 2]; title = L"Predicted $g_{t\phi}$",
        xlabel = L"r", ylabel = L"\cos^2\theta", zlabel = L"g_{t\phi}",
        azimuth = 1.3π, elevation = 0.2π)
CairoMakie.surface!(ax6a, collect(r_vals), collect(θ_vals), g_tp_true;
        colormap = :heat, colorrange = clims_tp)
CairoMakie.surface!(ax6b, collect(r_vals), collect(θ_vals), g_tp_nn;
        colormap = :viridis, colorrange = clims_tp)
CairoMakie.save("plots/final/6_gtp.png", fig6)
display(fig6)

# ── HELPER: relative error heatmap ───────────────────────────────────────────
function rel_err_fig(true_grid, nn_grid, title_str, fname)
    rel = (nn_grid .- true_grid) ./ (abs.(true_grid) .+ 1e-10)
    clim = max(maximum(abs.(rel)), 1e-6)

    fig = CairoMakie.Figure(size = (800, 600))
    gl  = CairoMakie.GridLayout(fig[1, 1])
    ax  = CairoMakie.Axis(gl[1, 1];
            title  = title_str,
            xlabel = L"r", ylabel = L"\theta",
            yticks = ([π/6, π/4, π/3, π/2],
                      [L"\pi/6", L"\pi/4", L"\pi/3", L"\pi/2"]))
    hm  = CairoMakie.heatmap!(ax, collect(r_vals), collect(θ_vals), rel;
            colormap = :RdBu, colorrange = (-clim, clim))
    Colorbar(gl[2, 1], hm; vertical = false, flipaxis = false, tellwidth = false)
    CairoMakie.save(fname, fig)
    display(fig)
    return fig
end

# ── 7. RELATIVE ERROR g_tt ───────────────────────────────────────────────────
rel_err_fig(g_tt_true, g_tt_nn,
    L"Relative Error $\delta g_{tt}$",
    "plots/final/7_gtt_relerr.png")

# ── 8. RELATIVE ERROR g_rr ───────────────────────────────────────────────────
rel_err_fig(g_rr_true, g_rr_nn,
    L"Relative Error $\delta g_{rr}$",
    "plots/final/8_grr_relerr.png")

# ── 9. RELATIVE ERROR g_tϕ ───────────────────────────────────────────────────
rel_err_fig(g_tp_true, g_tp_nn,
    L"Relative Error $\delta g_{t\phi}$",
    "plots/final/9_gtp_relerr.png")

c_vals = range(0.0, 1.0, length = 60)   # cosθ ∈ [0,1]

g_tp_true = [g_Kerr(@SVector [0.0, r, acos(c), 0.0])[1,4] for r in r_vals, c in c_vals]
g_tp_nn   = [nn_metric(r, acos(c))[1,4]                    for r in r_vals, c in c_vals]
clims_tp  = (min(minimum(g_tp_true), minimum(g_tp_nn)),
             max(maximum(g_tp_true), maximum(g_tp_nn)))

fig6 = CairoMakie.Figure(size = (1200, 500))
ax6a = CairoMakie.Axis3(fig6[1, 1]; title = L"True $g_{t\phi}$",
        xlabel = L"r", ylabel = L"\cos\theta", zlabel = L"g_{t\phi}",
        azimuth = 1.3π, elevation = 0.2π)
ax6b = CairoMakie.Axis3(fig6[1, 2]; title = L"Predicted $g_{t\phi}$",
        xlabel = L"r", ylabel = L"\cos\theta", zlabel = L"g_{t\phi}",
        azimuth = 1.3π, elevation = 0.2π)
CairoMakie.surface!(ax6a, collect(r_vals), collect(c_vals), g_tp_true;
        colormap = :heat, colorrange = clims_tp)
CairoMakie.surface!(ax6b, collect(r_vals), collect(c_vals), g_tp_nn;
        colormap = :viridis, colorrange = clims_tp)
CairoMakie.save("plots/final/6_gtp.png", fig6)
display(fig6)