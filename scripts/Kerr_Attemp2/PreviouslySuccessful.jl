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

include("PreviousNNFunctions.jl")
include("Orbit2Waveform.jl")
include("DissipationRicci.jl")
include("TrainingDataFunctions.jl")

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
tspan = (0, 500)
datasize = 4000
tsteps = range(tspan[1], tspan[2], length = datasize) 

rp_true = 4.285714285714286
ra_true = 10
θmin_true = π/3
a_true = 0.9
mass_ratio = 1e3

r_min, r_max = rp_true, ra_true

noise = 0

""" ************************
STEP 2: IMPORT TRAINING DATA
************************ """

waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Kerr_rp4.285714285714286_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.txt")
waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Kerr_rp4.285714285714286_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.txt")
true_solution = file2solution(tsteps,"input/solution_Kerr_rp4.285714285714286_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.txt")

waveform_real_faceon = waveform_real_ecc
waveform_imag_faceon = waveform_imag_ecc

# waveform_real_inclined = file2waveform(tsteps, "input/waveform_real_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.7853981633974483.txt")
# waveform_imag_inclined = file2waveform(tsteps, "input/waveform_imag_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.7853981633974483.txt")
# true_solution_inclined = file2solution(tsteps, "input/solution_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_0.7853981633974483.txt")

# waveform_real_inclined_2 = file2waveform(tsteps, "/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/input/waveform_real_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_1.0471975511965976.txt")
# waveform_imag_inclined_2 = file2waveform(tsteps, "/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/input/waveform_imag_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_1.0471975511965976.txt")
# true_solution_inclined_2 = file2solution(tsteps, "/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/input/solution_Kerr_rp6.666666666666667_ra10.0noise0.0_massRatio1000.0spin_0.9inclination_1.0471975511965976_time500_observertheta_1.0471975511965976.txt")

waveform_targets = [
    (
        name = "faceon",
        θ_obs = 0.0,
        ϕ_obs = 0.0,
        D = 1.0,
        true_h₊ = waveform_real_faceon,
        true_hₓ = waveform_imag_faceon
    )
    # (
    #     name = "inclined",
    #     θ_obs = π/4,
    #     ϕ_obs = 0.0,
    #     D = 1.0,
    #     true_h₊ = waveform_real_inclined,
    #     true_hₓ = waveform_imag_inclined
    # ),
    # (
    #     name = "inclined2",
    #     θ_obs = π/3,
    #     ϕ_obs = 0.0,
    #     D = 1.0,
    #     true_h₊ = waveform_real_inclined_2,
    #     true_hₓ = waveform_imag_inclined_2
    # )
]

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
    Dense(10, 6),
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

initialize_Conservative_NN(1e-3, [-0.1, 0.0, -0.1, -0.1, -0.1, 0.0], 0.3, 0.0)
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

t_init, h₊_init, hₓ_init, _, _ = nk_quadrupole_waveform(soln_nn; a = a_true, μ = 1.0, θ_obs = 0.0, ϕ_obs = 0.0, D = 1.0)
h₊_init = vec(h₊_init)
hₓ_init = vec(hₓ_init)

fig_h = Figure(size = (900, 450))
ax_h = CairoMakie.Axis(fig_h[1, 1];
    xlabel = L"t/M",
    ylabel = L"h(t)",
    title = L"h(t)"
)

lines!(ax_h, t_init, h₊_init; color = :blue, linewidth = 2, label = L"$h_+$ (Predicted)")
lines!(ax_h, t_init, hₓ_init; color = :red, linewidth = 2, label = L"$h_\times$ (Predicted)")
lines!(ax_h, tsteps, waveform_real_ecc; color = :blue, linewidth = 1, linestyle = :dash, label = L"$h_+$ (True)")
lines!(ax_h, tsteps, waveform_imag_ecc; color = :red, linewidth = 1, linestyle = :dash, label = L"$h_\times$ (True)")

axislegend(ax_h; position = :rt)
display(fig_h)

""" **************************
STEP 9: DEFINE A LOSS FUNCTION
************************** """

bias_history = []

rm("training_log.txt", force=true)

const θ_grid = range(π/6, 5π/6, length = 16)
const r_grid = range(1.2, 1.6, length = 96)

function waveform_mismatch(h₊_true, hₓ_true, h₊_pred, hₓ_pred; denom_floor = 1e-12)
    h₊_pred = vec(h₊_pred)
    hₓ_pred = vec(hₓ_pred)

    N = min(
        length(hₓ_pred),
        length(hₓ_true)
    )

    L_plus = sum(abs2, h₊_true[1:N] .- h₊_pred[1:N]) / 
            (sum(abs2, h₊_true[1:N]) + denom_floor)

    L_cross = sum(abs2, hₓ_true[1:N] .- hₓ_pred[1:N]) / 
            (sum(abs2, hₓ_true[1:N]) + denom_floor)

    return L_plus + L_cross, L_plus, L_cross
end

function grr_inverse_direct(r, θ, params)
    output, _ = NN_Conservative(
        [r / r_max, cos(θ)^2],
        params.conservative,
        NN_Conservative_state
    )

    A_rr = 1 + output[2] / r
    F_rr = 1 + output[6] / r

    return F_rr / A_rr
end

function soft_horizon_radius(θ, params; ϵ = 1e-6)
    weights = []
    radii = []

    for r in r_grid
        ginv = grr_inverse_direct(r, θ, params)

        w = 1 / (ginv^2 + ϵ)

        push!(weights, w)
        push!(radii, r)
    end

    return sum(weights .* radii) / sum(weights)
end

function horizon_boundary_condition(params; ϵ = 1e-6, σ_sign = 0.05)
    r_horizons = []
    null_terms = []
    sign_terms = []
    min_abs_terms = []

    r_lo = first(r_grid)
    r_hi = last(r_grid)

    for θ in θ_grid
        r_soft = soft_horizon_radius(θ, params; ϵ = ϵ)
        
        g_soft = grr_inverse_direct(r_soft, θ, params)
        null_term = abs2(g_soft)

        g_lo = grr_inverse_direct(r_lo, θ, params)
        g_hi = grr_inverse_direct(r_hi, θ, params)

        sign_term = softplus(g_lo / σ_sign)^2 
                  + softplus(-g_hi / σ_sign)^2

        gvals = [grr_inverse_direct(r, θ, params) for r in r_grid]
        min_abs_g = minimum(abs.(gvals))

        push!(r_horizons, r_soft)
        push!(null_terms, null_term)
        push!(sign_terms, sign_term)
        push!(min_abs_terms, min_abs_g)
    end
    
    r_mean = mean(r_horizons)

    variance_term = mean((r_horizons .- r_mean).^2)
    continuity_term = mean(diff(r_horizons).^2)

    null_surface_term = mean(null_terms)
    sign_change_term = mean(sign_terms)
    min_abs_g_mean = mean(min_abs_terms)

    val_bc = variance_term + continuity_term + null_surface_term + sign_change_term

    return val_bc, null_surface_term, sign_change_term, variance_term, continuity_term, r_mean, min_abs_g_mean
end

function horizon_diagnostics(params)
    val_bc, val_null, val_sign, variance_term, continuity_term, rH_mean, min_abs_g =
        horizon_boundary_condition(params)

    r_horizons = [soft_horizon_radius(θ, params) for θ in θ_grid]

    grr_soft_vals = [
        grr_inverse_direct(r_horizons[k], θ_grid[k], params)
        for k in eachindex(θ_grid)
    ]

    g_lo_vals = [grr_inverse_direct(first(r_grid), θ, params) for θ in θ_grid]
    g_hi_vals = [grr_inverse_direct(last(r_grid), θ, params) for θ in θ_grid]

    sign_ok = [
        g_lo_vals[k] * g_hi_vals[k] < 0
        for k in eachindex(θ_grid)
    ]

    return (
        val_bc          = ForwardDiff.value(val_bc),
        val_null        = ForwardDiff.value(val_null),
        val_sign        = ForwardDiff.value(val_sign),
        variance_term   = ForwardDiff.value(variance_term),
        continuity_term = ForwardDiff.value(continuity_term),
        rH_mean         = ForwardDiff.value(rH_mean),
        rH_std          = std(ForwardDiff.value.(r_horizons)),
        min_abs_g       = ForwardDiff.value(min_abs_g),
        grr_soft_rms    = sqrt(mean(abs2, ForwardDiff.value.(grr_soft_vals))),
        grr_soft_min    = minimum(ForwardDiff.value.(grr_soft_vals)),
        grr_soft_max    = maximum(ForwardDiff.value.(grr_soft_vals)),
        g_lo_mean       = mean(ForwardDiff.value.(g_lo_vals)),
        g_hi_mean       = mean(ForwardDiff.value.(g_hi_vals)),
        sign_success    = count(sign_ok),
        sign_total      = length(sign_ok)
    )
end
 
function killing_horizon_loss(params)
    Ω_vals = []
    χ_vals = []

    for θᵢ in θ_grid
        r_H = soft_horizon_radius(θᵢ, params)

        sv = @SVector [0.0, r_H, θᵢ, 0.0]
        output, _ = NN_Conservative([r_H / r_max, cos(θᵢ)^2],
                                    params.conservative, NN_Conservative_state)
        g = g_NeuralNetwork(sv, output)

        g_tt, g_tphi, g_pp = g[1,1], g[1,4], g[4,4]

        Ω = -g_tphi / g_pp
        χ = g_tt + 2Ω*g_tphi + Ω^2 * g_pp

        push!(Ω_vals, Ω)
        push!(χ_vals, χ)
    end

    L_Ω = mean((Ω_vals .- mean(Ω_vals)).^2)
    L_Χ = mean(abs2.(χ_vals))

    return L_Ω + L_Χ
end

function make_waveforms_bank(pred_soln, waveform_targets; a = a_true, μ = 1.0)
    bank = []

    for target in waveform_targets
        _, pred_h₊, pred_hₓ, _, _ = 
        nk_quadrupole_waveform(
            pred_soln; 
            a = a,
            μ = μ, 
            θ_obs = target.θ_obs,
            ϕ_obs = target.ϕ_obs, 
            D = target.D
        )

        push!(bank, merge(target, (
            pred_h₊ = vec(pred_h₊),
            pred_hₓ = vec(pred_hₓ)
        )))
    end
    return bank
end

function loss(NN_params; saveat = tsteps, ricci_weight = 0.0)
    ricci_loss = 0.0
    val_wave = 0.0
    view_losses = []
    
    function metric(x)
        r_val = x[2]
        θ_val = x[3]
        output, _ = NN_Conservative([r_val / r_max, (cos(θ_val))^2],
                                    NN_params.conservative, NN_Conservative_state)
        return g_NeuralNetwork(x, output)
    end

    # -----------------------------------------------
    # METRIC LOSS  
    # -----------------------------------------------
    r_check = range(2.5, r_max, length = 32)
    θ_check = range(π/15, 14π/15, length = 32)
    
    for radius in r_check
        for angle in θ_check
            input_state = @SVector [0, radius, angle, 0.0]
            Riemann_Tensor = Riemann(metric, input_state)
            Ricci_covariant = RicciTensor(Riemann_Tensor)
            ricci_loss += sum(abs2, Ricci_covariant)
        end
    end
    
    ricci_loss = ricci_loss / (length(r_check) * length(θ_check))

    # -----------------------------------------------
    # WAVEFORM LOSS  
    # -----------------------------------------------
    tspan = (saveat[1], saveat[end])
    u0_local = make_u0(rp_true, ra_true, θmin_true)
    prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)
    pred_soln = Array(solve(prob_pred, Tsit5(); saveat = saveat, dt = dt, adaptive = false, verbose = false))

    N_expected = length(saveat)
    L2_metric_loss = 0
    
    if size(pred_soln, 2) < N_expected || any(isnan, pred_soln) || any(isinf, pred_soln)
        dummy_wave = zeros(N_expected)
        bad_loss = 1e10
        return (bad_loss, bad_loss, 0.0, 0.0, 0.0, ForwardDiff.value(L2_metric_loss)),
                dummy_wave, dummy_wave, pred_soln
    end

    for target in waveform_targets
        _, h₊_pred_view, hₓ_pred_view, _, _ = 
            nk_quadrupole_waveform(
                pred_soln; 
                a = a_true,
                μ = 1.0,
                θ_obs = target.θ_obs,
                ϕ_obs = target.ϕ_obs,
                D = target.D
            )

        L_view, L_plus, L_cross = 
                                waveform_mismatch(
                                    target.true_h₊,
                                    target.true_hₓ,
                                    h₊_pred_view,
                                    hₓ_pred_view
                                )

        val_wave += L_view

        push!(view_losses, (
            name = target.name,
            L_total = L_view,
            L_plus = L_plus,
            L_cross = L_cross
        ))
    end

    val_wave /= length(waveform_targets)

    val_ricci = ricci_loss

    val_bc, val_null, val_sign, variance_term, continuity_term, rH_mean, min_abs_g = horizon_boundary_condition(NN_params)

    val_killing = killing_horizon_loss(NN_params)

    total_loss = val_wave + val_ricci + val_bc + val_killing

    plot_bank = make_waveforms_bank(pred_soln, waveform_targets; a = a_true, μ = 1.0)

    h₊_pred = plot_bank[1].pred_h₊
    hₓ_pred = plot_bank[1].pred_hₓ

    return (
            total_loss,
            ForwardDiff.value(val_wave),
            ForwardDiff.value(val_ricci),
            ForwardDiff.value(val_killing),
            ForwardDiff.value(val_bc),
            ForwardDiff.value(L2_metric_loss)
        ),
        h₊_pred,
        hₓ_pred,
        pred_soln 
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
    CairoMakie.lines!(ax, collect(iter), bound_loss;  linewidth = 2, label = "Boundary Condition", linestyle = :dash)
    CairoMakie.lines!(ax, collect(iter), flatn_loss; linewidth = 2, label = "Killing Horizon", linestyle = :dash)
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

checkpoint_dir = "checkpoints"
mkpath(checkpoint_dir)

function save_checkpoint(p, iter, increment, current_loss; tag = nothing)
    NN_params_checkpoint = deepcopy(ForwardDiff.value.(p))

    filename = isnothing(tag) ?
        joinpath(checkpoint_dir, @sprintf("NN_params_iter_%06d.jld2", iter)) :
        joinpath(checkpoint_dir, "$(tag).jld2")

    @save filename NN_params_checkpoint iter increment current_loss r_max r_min a_true rp_true ra_true θmin_true mass_ratio

    open(joinpath(checkpoint_dir, "checkpoint_log.txt"), "a") do io 
        println(io, "Saved checkpoint at iteration ", iter,
                    ", increment ", increment,
                    ", loss = ", current_loss,
                    ", file = ", filename)
    end

    println("Saved checkpoint: ", filename)
end

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
        (loss_val, v_wave, v_ricci, v_flat, v_bound, v_metric), pred_h₊, pred_hₓ, pred_soln = tmp_loss(p)
        return loss_val  # Return only the loss value for gradient computation
    end

    function build_dashboard(p, pred_soln, waveforms_bank_current,
                            tsteps, N, startPoint, losses, waveform_loss,
                            ricci_losses, flatness_loss, bound_loss,
                            metric_loss, loss_val)

        fig = CairoMakie.Figure(size = (1800, 1600))

        # ── Row 1: diagnostics ──────────────────────────────────────────
        plot_waveform!(fig[1, 1], tsteps, waveforms_bank_current,
            startPoint, N
        )
        plot_geodesic_3d!(fig[1, 2], pred_soln, true_solution)
        plot_loss!(fig[1, 3], losses, waveform_loss, ricci_losses, flatness_loss, bound_loss, metric_loss)

        # ── Rows 2–3: metric surfaces ───────────────────────────────────
        plot_metric_components!(fig[2:3, :], p, 3, r_max, a_true)   # pass row range

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
        
        metric_labels = [
            L"\mathrm{NN}_1: g_{tt}",
            L"\mathrm{NN}_2: A_{rr}",
            L"\mathrm{NN}_3: g_{\theta\theta}",
            L"\mathrm{NN}_4: g_{\phi\phi}",
            L"\mathrm{NN}_5: g_{t\phi}",
            L"\mathrm{NN}_6: F_{rr}"
        ]

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

        (loss_val, v_wave, v_ricci, v_flat, v_bound, v_metric), pred_h₊, pred_hₓ, pred_soln = tmp_loss(p)
        hdiag = horizon_diagnostics(p)

        waveforms_bank_current = 
                make_waveforms_bank(pred_soln, waveform_targets; a = a_true, μ = 1.0)
        
        global NN_params, best_loss, best_NN_params

        current_loss = ForwardDiff.value(loss_val)
        global_iter = length(losses) + 1

        NN_params = deepcopy(ForwardDiff.value.(p))

        if current_loss < best_loss
            best_loss = current_loss
            best_NN_params = deepcopy(ForwardDiff.value.(p))
            @save "best_NN_params.jld2" best_NN_params best_loss
        end

        if global_iter % 50 == 0
            save_checkpoint(p, global_iter, i, current_loss)
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
            @printf(io, "  Boundary Condition Loss : %.6e\n", v_bound)
            @printf(io, "  Killing Horizon Loss    : %.6e\n", v_flat)

            @printf(io, "    BC Null Surface       : %.6e\n", hdiag.val_null)
            @printf(io, "    BC Sign Change        : %.6e\n", hdiag.val_sign)
            @printf(io, "    BC Variance           : %.6e\n", hdiag.variance_term)
            @printf(io, "    BC Continuity         : %.6e\n", hdiag.continuity_term)
            @printf(io, "    rH mean               : %.6f\n", hdiag.rH_mean)
            @printf(io, "    rH std                : %.6e\n", hdiag.rH_std)
            @printf(io, "    min |g^rr| on grid    : %.6e\n", hdiag.min_abs_g)
            @printf(io, "    rms g^rr(r_soft)      : %.6e\n", hdiag.grr_soft_rms)
            @printf(io, "    range g^rr(r_soft)    : [%.6e, %.6e]\n",
                    hdiag.grr_soft_min, hdiag.grr_soft_max)
            @printf(io, "    mean g^rr(r_lo)       : %.6e\n", hdiag.g_lo_mean)
            @printf(io, "    mean g^rr(r_hi)       : %.6e\n", hdiag.g_hi_mean)
            @printf(io, "    sign changes          : %d / %d\n", hdiag.sign_success, hdiag.sign_total)
            
            println(io, "─"^50)
            println(io, "Componentwise metric diagnostics:")
            
            r_table = [6.0, 8.0, 12.0, 16.0, 20.0]
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

        N = min(
            length(tsteps),
            length(waveform_real_ecc),
            length(waveform_imag_ecc),
            length(pred_h₊),
            length(pred_hₓ)
        )

        startPoint = 1

        dashboard_every = 5

        if global_iter % dashboard_every == 0
            fig = build_dashboard(p, pred_soln, waveforms_bank_current,
                        tsteps, N, startPoint,
                        losses, waveform_loss, ricci_losses,
                        flatness_losses, boundOrbit_losses,
                        metric_losses, loss_val)

            mkpath("plots/dashboard")
            save(joinpath("plots/dashboard", @sprintf("dashboard_iter_%06d.png", global_iter)), fig)

            try
                display(fig)
            catch e
                @warn "Dashboard display failed, but optimization will continue." exception=e
            end
        end

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
                callback=opt_callback, allow_f_increases=false, maxiters=300)
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
                callback=opt_callback, allow_f_increases=false, maxiters=200)
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
                                    allow_f_increases=false, maxiters = 100)
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
                                    allow_f_increases=false, maxiters = 90)
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
                                    allow_f_increases=false, maxiters = 80)
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
                                    allow_f_increases=false, maxiters = 60)
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

if length(losses) > 0
    final_iter = length(losses)
    final_loss = ForwardDiff.value(losses[end])

    save_checkpoint(
        NN_params,
        final_iter,
        optimization_increments[end],
        final_loss;
        tag = @sprintf("NN_params_final_iter_%06d", final_iter)
    )
end

println("Training complete.")
println("Final optimizer parameters remain in NN_params.")
println("Best-so-far parameters are saved separately in best_NN_params.jld2.")

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