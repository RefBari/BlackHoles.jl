using GeometryBasics: Point2f, Polygon
using CairoMakie
using StaticArrays
using JLD2
using ForwardDiff
using LaTeXStrings

# ============================================================
# Load best NN parameters
# ============================================================

@load "best_NN_params.jld2" best_NN_params best_loss
NN_params = best_NN_params
println("Loaded best loss = ", best_loss)

mkpath("plots/final")

# ============================================================
# Optional: publication-ish theme
# ============================================================

CairoMakie.set_theme!(
    fontsize = 18,
    fonts = (;
        regular = "Latin Modern Roman",
        bold = "Latin Modern Roman",
        italic = "Latin Modern Roman"
    )
)

# ============================================================
# Plot saving helper
# ============================================================

function save_highres(path_base, fig; px_per_unit = 3)
    CairoMakie.save(path_base * ".png", fig; px_per_unit = px_per_unit)
    CairoMakie.save(path_base * ".pdf", fig)
end

# Convert ForwardDiff / dual-ish numbers to Float64 for plotting
tofloat(x) = Float64(ForwardDiff.value(x))
tofloat_array(A) = Float64.(ForwardDiff.value.(A))

# ============================================================
# LaTeX label helpers
# ============================================================

function θ_tex(θ)
    if isapprox(θ, π/8; atol = 1e-12)
        return "\\pi/8"
    elseif isapprox(θ, π/4; atol = 1e-12)
        return "\\pi/4"
    elseif isapprox(θ, π/2; atol = 1e-12)
        return "\\pi/2"
    else
        return string(round(θ, digits = 3))
    end
end

θ_label(θ) = latexstring("\\theta = " * θ_tex(θ))
r_label(r) = latexstring("r = $(Int(round(r)))")
point_label(r, θ) = latexstring("r = $(Int(round(r)))" * ",\\; \\theta = " * θ_tex(θ))

function numeric_latex(x; sigdigits = 3)
    s = string(round(x, sigdigits = sigdigits))
    return latexstring("\\mathrm{" * s * "}")
end

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

function ricci_tensor_at(r, θ)
    x = @SVector [0.0, r, θ, 0.0]
    Riem = Riemann(pred_metric, x)
    Ric = RicciTensor(Riem)
    return tofloat_array(Ric)
end

function ricci_residual_at(r, θ)
    Ric = ricci_tensor_at(r, θ)
    return sum(abs2, Ric)
end

function metric_error_at(r, θ, i, j; relative = true, eps = 1e-12)
    x = @SVector [0.0, r, θ, 0.0]

    g_pred = tofloat(pred_metric(x)[i, j])
    g_true = tofloat(true_metric(x)[i, j])

    if relative
        return (g_pred - g_true) / (abs(g_true) + eps)
    else
        return g_pred - g_true
    end
end

# ============================================================
# Sample points
# ============================================================

r_samples = [5.0, 10.0, 18.0]
θ_samples = [π/8, π/4, π/2]

heatmap_points = [
    (6.0,  π/2), (6.0,  π/4), (6.0,  π/8),
    (12.0, π/2), (12.0, π/4), (12.0, π/8),
    (18.0, π/2), (18.0, π/4), (18.0, π/8)
]

# ============================================================
# Polar-sector helpers
# ============================================================

function radial_edges_from_centers(r_centers; rmin = 5.0, rmax = 20.0)
    mids = [(r_centers[i] + r_centers[i+1]) / 2 for i in 1:length(r_centers)-1]
    return [rmin; mids; rmax]
end

function angular_edges_from_centers(θ_centers; θmin = 0.0, θmax = π/2)
    mids = [(θ_centers[i] + θ_centers[i+1]) / 2 for i in 1:length(θ_centers)-1]
    return [θmin; mids; θmax]
end

function sector_polygon(r1, r2, θ1, θ2; n = 100)
    θ_outer = range(θ1, θ2, length = n)
    θ_inner = range(θ2, θ1, length = n)

    pts = Point2f[]

    for θ in θ_outer
        push!(pts, Point2f(r2*cos(θ), r2*sin(θ)))
    end

    for θ in θ_inner
        push!(pts, Point2f(r1*cos(θ), r1*sin(θ)))
    end

    return Polygon(pts)
end

function polar_sector_plot!(ax, r_centers, θ_centers, values;
                            rmin = 5.0,
                            rmax = 20.0,
                            θmin = 0.0,
                            θmax = π/2,
                            title_str = L"",
                            colormap = :viridis,
                            diverging = false,
                            log10_color = false)

    r_edges = radial_edges_from_centers(r_centers; rmin = rmin, rmax = rmax)
    θ_edges = angular_edges_from_centers(θ_centers; θmin = θmin, θmax = θmax)

    polys = Polygon[]
    cs = Float64[]

    for ir in 1:length(r_centers)
        for it in 1:length(θ_centers)
            push!(
                polys,
                sector_polygon(
                    r_edges[ir],
                    r_edges[ir+1],
                    θ_edges[it],
                    θ_edges[it+1]
                )
            )

            val = values[ir, it]

            if log10_color
                val = log10(abs(val) + 1e-16)
            end

            push!(cs, Float64(val))
        end
    end

    if diverging
        m = maximum(abs.(cs))
        if !isfinite(m) || m == 0.0
            m = 1.0
        end
        crange = (-m, m)
    else
        cmin = minimum(cs)
        cmax = maximum(cs)
        if !isfinite(cmin) || !isfinite(cmax) || cmin == cmax
            cmin, cmax = 0.0, 1.0
        end
        crange = (cmin, cmax)
    end

    plt = CairoMakie.poly!(
        ax,
        polys;
        color = cs,
        colormap = colormap,
        colorrange = crange,
        strokecolor = :white,
        strokewidth = 2
    )

    # Guide arcs
    ϕ = range(θmin, θmax, length = 350)
    for r in r_edges
        CairoMakie.lines!(
            ax,
            r .* cos.(ϕ),
            r .* sin.(ϕ);
            color = (:black, 0.22),
            linewidth = 1
        )
    end

    # Guide rays
    for θ in θ_edges
        CairoMakie.lines!(
            ax,
            [0.0, rmax*cos(θ)],
            [0.0, rmax*sin(θ)];
            color = (:black, 0.22),
            linewidth = 1
        )
    end

    # Radius labels
    for r in r_centers
        CairoMakie.text!(
            ax,
            r*cos(0.045),
            r*sin(0.045);
            text = r_label(r),
            fontsize = 14,
            color = :black,
            align = (:left, :bottom)
        )
    end

    # Angle labels
    for θ in θ_centers
        CairoMakie.text!(
            ax,
            1.07*rmax*cos(θ),
            1.07*rmax*sin(θ);
            text = θ_label(θ),
            fontsize = 15,
            color = :black,
            align = (:center, :center)
        )
    end

    ax.title = title_str
    ax.aspect = CairoMakie.DataAspect()

    CairoMakie.hidespines!(ax)
    CairoMakie.hidexdecorations!(ax; grid = false)
    CairoMakie.hideydecorations!(ax; grid = false)

    return plt
end

# ============================================================
# Plot 1: Ricci residual ||R_{μν}||²
# ============================================================

ricci_residual_vals = [
    ricci_residual_at(r, θ)
    for r in r_samples, θ in θ_samples
]

fig1 = CairoMakie.Figure(size = (1000, 780))
ax1 = CairoMakie.Axis(fig1[1, 1])

pl1 = polar_sector_plot!(
    ax1,
    r_samples,
    θ_samples,
    ricci_residual_vals;
    rmin = 5.0,
    rmax = 20.0,
    θmin = 0.0,
    θmax = π/2,
    title_str = L"\mathrm{Ricci\ residual:}\ \log_{10}\Vert R_{\mu\nu}\Vert^2",
    colormap = :viridis,
    diverging = false,
    log10_color = true
)

CairoMakie.Colorbar(
    fig1[1, 2],
    pl1;
    label = L"\log_{10}\left(\Vert R_{\mu\nu}\Vert^2\right)"
)

display(fig1)
save_highres("plots/final/1_ricci_residual_polar", fig1)

# ============================================================
# Plot 2: Relative errors in 5 metric components
# ============================================================

metric_components = [
    ((1, 1), "g_{tt}"),
    ((2, 2), "g_{rr}"),
    ((3, 3), "g_{\\theta\\theta}"),
    ((4, 4), "g_{\\phi\\phi}"),
    ((1, 4), "g_{t\\phi}")
]

fig2 = CairoMakie.Figure(size = (2100, 1000))

for (k, ((i, j), tex)) in enumerate(metric_components)
    row = k <= 3 ? 1 : 2
    col = k <= 3 ? k : k - 3

    ax_col = 2*col - 1
    cb_col = 2*col

    ax = CairoMakie.Axis(fig2[row, ax_col])

    vals = [
        metric_error_at(r, θ, i, j; relative = true)
        for r in r_samples, θ in θ_samples
    ]

    pl = polar_sector_plot!(
        ax,
        r_samples,
        θ_samples,
        vals;
        rmin = 5.0,
        rmax = 20.0,
        θmin = 0.0,
        θmax = π/2,
        title_str = latexstring("\\mathrm{Relative\\ error\\ in}\\ " * tex),
        colormap = :RdBu,
        diverging = true,
        log10_color = false
    )

    CairoMakie.Colorbar(
        fig2[row, cb_col],
        pl;
        label = latexstring("\\delta " * tex)
    )
end

CairoMakie.Label(
    fig2[0, :],
    L"\mathrm{Metric\ component\ errors:}\quad \delta g_{\mu\nu}=\frac{g^{\rm pred}_{\mu\nu}-g^{\rm true}_{\mu\nu}}{|g^{\rm true}_{\mu\nu}|+\epsilon}";
    fontsize = 25,
    font = :bold
)

display(fig2)
save_highres("plots/final/2_metric_errors_polar", fig2)

# ============================================================
# Plot 3: 9 separate 4x4 Ricci tensor heatmaps
# ============================================================

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
                color = abs(val) > 0.55*maxabs_ricci ? :white : :black
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

# ============================================================
# Plot 4: 10 independent Ricci tensor components
# ============================================================

ricci_components = [
    ((1, 1), "R_{tt}"),
    ((2, 2), "R_{rr}"),
    ((3, 3), "R_{\\theta\\theta}"),
    ((4, 4), "R_{\\phi\\phi}"),
    ((1, 2), "R_{tr}"),
    ((1, 3), "R_{t\\theta}"),
    ((1, 4), "R_{t\\phi}"),
    ((2, 3), "R_{r\\theta}"),
    ((2, 4), "R_{r\\phi}"),
    ((3, 4), "R_{\\theta\\phi}")
]

fig4 = CairoMakie.Figure(size = (2450, 1100))

for (k, ((i, j), tex)) in enumerate(ricci_components)
    row = k <= 5 ? 1 : 2
    col = k <= 5 ? k : k - 5

    ax_col = 2*col - 1
    cb_col = 2*col

    ax = CairoMakie.Axis(fig4[row, ax_col])

    vals = [
        ricci_tensor_at(r, θ)[i, j]
        for r in r_samples, θ in θ_samples
    ]

    pl = polar_sector_plot!(
        ax,
        r_samples,
        θ_samples,
        vals;
        rmin = 5.0,
        rmax = 20.0,
        θmin = 0.0,
        θmax = π/2,
        title_str = latexstring(tex),
        colormap = :RdBu,
        diverging = true,
        log10_color = false
    )

    CairoMakie.Colorbar(
        fig4[row, cb_col],
        pl;
        label = latexstring(tex)
    )
end

CairoMakie.Label(
    fig4[0, :],
    L"\mathrm{Independent\ Ricci\ tensor\ components}\ R_{\mu\nu}";
    fontsize = 26,
    font = :bold
)

display(fig4)
save_highres("plots/final/4_ricci_components_polar", fig4)

println("Done. Saved high-resolution PNG and PDF diagnostics to plots/final/")