using CairoMakie
using JLD2
using StaticArrays
using ForwardDiff

# ============================================================
# Load best NN params
# ============================================================

@load "best_NN_params.jld2" best_NN_params best_loss
NN_params = best_NN_params
println("Loaded best loss = ", best_loss)

mkpath("plots/final")

function save_highres(path_base, fig; px_per_unit = 3)
    CairoMakie.save(path_base * ".png", fig; px_per_unit = px_per_unit)
    CairoMakie.save(path_base * ".pdf", fig)
end

tofloat(x) = Float64(ForwardDiff.value(x))
tofloat_array(A) = Float64.(ForwardDiff.value.(A))

# ============================================================
# Predicted solution using saved NN params
# ============================================================

u0_local = make_u0(rp_true, ra_true, θmin_true; a = a_true)

prob_pred = ODEProblem(ODE_model, u0_local, tspan, NN_params)

pred_soln = Array(
    solve(prob_pred, Tsit5();
          saveat = tsteps,
          dt = dt,
          adaptive = false)
)

# ============================================================
# Waveforms
# ============================================================

pred_waveform_real, pred_waveform_imag = compute_waveform(dt_data, pred_soln, mass_ratio)
true_waveform_real, true_waveform_imag = compute_waveform(dt_data, true_solution, mass_ratio)

L2_waveform = norm(pred_waveform_real .- true_waveform_real)
println("Final waveform L2 error = ", L2_waveform)

# ============================================================
# Metric helpers
# ============================================================

function pred_metric(x)
    r = x[2]
    θ = x[3]

    out, _ = NN_Conservative(
        [r / r_max, cos(θ)^2],
        NN_params.conservative,
        NN_Conservative_state
    )

    return g_NeuralNetwork(x, out)
end

function true_metric(x)
    return g_Kerr(x; M = 1.0, a = a_true)
end

function metric_component_grid(i, j, r_vals, θ_vals; mode = :pred, relative = false, eps = 1e-12)
    Z = zeros(length(r_vals), length(θ_vals))

    for (ir, r) in enumerate(r_vals)
        for (it, θ) in enumerate(θ_vals)
            x = @SVector [0.0, r, θ, 0.0]

            gpred = tofloat(pred_metric(x)[i, j])
            gtrue = tofloat(true_metric(x)[i, j])

            if mode == :pred
                Z[ir, it] = gpred
            elseif mode == :true
                Z[ir, it] = gtrue
            elseif mode == :error
                if relative
                    Z[ir, it] = (gpred - gtrue) / (abs(gtrue) + eps)
                else
                    Z[ir, it] = gpred - gtrue
                end
            end
        end
    end

    return Z
end

# ============================================================
# Plot 1: Orbit + waveform
# ============================================================

fig1 = CairoMakie.Figure(size = (1600, 700))

# Orbit panel
plot_geodesic_3d!(fig1[1, 1], pred_soln, true_solution; a = a_true)

# Waveform panel
axw = CairoMakie.Axis(
    fig1[1, 2];
    title = "Waveform Comparison",
    xlabel = "t",
    ylabel = "h₊"
)

N = min(length(tsteps), length(pred_waveform_real), length(true_waveform_real))

CairoMakie.lines!(axw, tsteps[1:N], true_waveform_real[1:N];
    linewidth = 2,
    label = "True")

CairoMakie.lines!(axw, tsteps[1:N], pred_waveform_real[1:N];
    linewidth = 2,
    linestyle = :dash,
    label = "Predicted")

CairoMakie.axislegend(axw, position = :rb)

CairoMakie.Label(
    fig1[0, :],
    "Predicted orbit and waveform  |  best loss = $(round(best_loss, sigdigits=4))";
    fontsize = 24,
    font = :bold
)

display(fig1)
save_highres("plots/final/orbit_and_waveform", fig1)

# ============================================================
# Plot 2: r(t) comparison (optional but useful)
# ============================================================

fig_r = CairoMakie.Figure(size = (900, 550))
axr = CairoMakie.Axis(
    fig_r[1, 1];
    title = L"r(t)",
    xlabel = L"t",
    ylabel = L"r"
)

CairoMakie.lines!(axr, tsteps, true_solution[2, :];
    linewidth = 2,
    label = "True")

CairoMakie.lines!(axr, tsteps, pred_soln[2, :];
    linewidth = 2,
    linestyle = :dash,
    label = "Predicted")

CairoMakie.axislegend(axr, position = :rt)
display(fig_r)
save_highres("plots/final/r_vs_t", fig_r)

# ============================================================
# Plot 3: Metric component surfaces
# ============================================================

r_vals = range(6, 12, length = 10)
θ_vals = range(π/32, π/2, length = 10)

metric_components = [
    ((1,1), "g_{tt}"),
    ((2,2), "g_{rr}"),
    ((3,3), "g_{\\theta\\theta}"),
    ((4,4), "g_{\\phi\\phi}"),
    ((1,4), "g_{t\\phi}")
]

fig2 = CairoMakie.Figure(size = (1800, 2400))

for (row, ((i, j), label)) in enumerate(metric_components)
    Z_pred = metric_component_grid(i, j, r_vals, θ_vals; mode = :pred)
    Z_true = metric_component_grid(i, j, r_vals, θ_vals; mode = :true)
    Z_err  = metric_component_grid(i, j, r_vals, θ_vals; mode = :error, relative = true)

    # color range for pred/true together
    zmin = min(minimum(Z_pred), minimum(Z_true))
    zmax = max(maximum(Z_pred), maximum(Z_true))
    if zmin == zmax
        zmin -= 1
        zmax += 1
    end

    # symmetric color range for error
    errmax = maximum(abs.(Z_err))
    if errmax == 0
        errmax = 1.0
    end

    # Predicted
    ax1 = CairoMakie.Axis(
        fig2[row, 1];
        title = "Predicted $label",
        xlabel = "θ",
        ylabel = "r"
    )
    hm1 = CairoMakie.heatmap!(ax1, θ_vals, r_vals, Z_pred;
        colormap = :viridis,
        colorrange = (zmin, zmax))
    CairoMakie.Colorbar(fig2[row, 2], hm1)

    # True
    ax2 = CairoMakie.Axis(
        fig2[row, 3];
        title = "True $label",
        xlabel = "θ",
        ylabel = "r"
    )
    hm2 = CairoMakie.heatmap!(ax2, θ_vals, r_vals, Z_true;
        colormap = :viridis,
        colorrange = (zmin, zmax))
    CairoMakie.Colorbar(fig2[row, 4], hm2)

    # Relative error
    ax3 = CairoMakie.Axis(
        fig2[row, 5];
        title = latexstring("\\mathrm{Relative\\ error\\ in}\\ " * label),
        xlabel = L"r",
        ylabel = L"\theta"
    )
    hm3 = CairoMakie.heatmap!(ax3, r_vals, θ_vals, Z_err;
        colormap = :RdBu,
        colorrange = (-errmax, errmax))
    CairoMakie.Colorbar(fig2[row, 6], hm3)
end

CairoMakie.Label(
    fig2[0, :],
    "Predicted metric vs true Kerr metric";
    fontsize = 26,
    font = :bold
)

display(fig2)
save_highres("plots/final/metric_components_comparison", fig2)

println("Done.")
println("Saved:")
println("  plots/final/orbit_and_waveform.png/.pdf")
println("  plots/final/r_vs_t.png/.pdf")
println("  plots/final/metric_components_comparison.png/.pdf")

# ============================================================
# Plot 3: 9 separate 4x4 Ricci tensor heatmaps
# ============================================================

heatmap_points = [
    (3.0,  π/2), (3.0,  π/4), (3.0,  π/8),
    (6.0,  π/2), (6.0,  π/4), (6.0,  π/8),
    (12.0, π/2), (12.0, π/4), (12.0, π/8)
]

ricci_mats = [ricci_tensor_at(r, θ) for (r, θ) in heatmap_points]

maxabs_ricci = maximum(abs.([
    Ric[μ, ν]
    for Ric in ricci_mats
    for μ in 1:4
    for ν in 1:4
]))

if !isfinite(maxabs_ricci) || maxabs_ricci == 0.0
    maxabs_ricci = 1.0
end

labels4 = [L"t", L"r", L"\theta", L"\phi"]

fig3 = CairoMakie.Figure(size = (1550, 1400))

hm_ref = nothing

for (idx, ((r, θ), Ric)) in enumerate(zip(heatmap_points, ricci_mats))
    row = div(idx - 1, 3) + 1
    col = mod(idx - 1, 3) + 1

    ax = CairoMakie.Axis(
        fig3[row, col];
        title = point_label(r, θ),
        xticks = (1:4, labels4),
        yticks = (1:4, labels4),
        xlabel = L"\nu",
        ylabel = L"\mu"
    )

    hm_ref = CairoMakie.heatmap!(
        ax,
        1:4,
        1:4,
        Ric;
        colormap = :RdBu,
        colorrange = (-maxabs_ricci, maxabs_ricci)
    )

    for μ in 1:4
        for ν in 1:4
            val = Ric[μ, ν]

            CairoMakie.text!(
                ax,
                ν,
                μ;
                text = numeric_latex(val; sigdigits = 3),
                align = (:center, :center),
                fontsize = 11,
                color = abs(val) > 0.55 * maxabs_ricci ? :white : :black
            )
        end
    end
end

CairoMakie.Colorbar(
    fig3[:, 4],
    hm_ref;
    label = L"R_{\mu\nu}"
)

CairoMakie.Label(
    fig3[0, :],
    L"\mathrm{Full\ Ricci\ tensor\ residuals}\ R_{\mu\nu}\ \mathrm{at\ nine\ sample\ points}";
    fontsize = 26,
    font = :bold
)

display(fig3)
save_highres("plots/final/3_ricci_tensor_9_heatmaps", fig3)