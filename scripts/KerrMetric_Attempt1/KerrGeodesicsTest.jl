using KerrGeodesics
using CairoMakie
using OrdinaryDiffEq
using NLsolve
using ForwardDiff

# Orbital parameters
a    = 0.9
rp   = 2.9
ra   = 4  
θmin = π/4               # equatorial

M    = 1.0
tmax = 400.0
dt   = 0.1

# ---------------------
# 1. Ref's Geodesic
# ---------------------
u0, E, Lz, Q = u0_from_geometry(rp, ra, θmin; M = M, a = a, μ = 1.0)
sol_ref = solve_kerr(u0; M = M, a = a, λmax = tmax, saveat = dt)
soln = Array(sol_ref)

t_mine = soln[1, :]
r_mine = soln[2, :]
θ_mine = soln[3, :]
ϕ_mine = soln[4, :]

# ---------------------
# 2. KerrGeodesics.jl
# ---------------------
p = 2 * rp * ra / (rp + ra)
e = (ra - rp) / (ra + rp)
x_inc = sin(θmin)

EMRI_info = kerr_geo_emri(a, p, e, x_inc)
t_fn, r_fn, θ_fn, ϕ_fn = EMRI_info.Trajectory

λ_test = range(0, 50, length = 10000)
t_scan = [t_fn(λ) for λ in λ_test]

idx_end = findfirst(t_scan .>= tmax)
if isnothing(idx_end)
    @warn "tmax not reached! increase λ upper bound above 50.0"
    idx_end = length(λ_test)
end
λ_max = λ_test[idx_end]

N_λ    = 20_000
λ_dense = range(0.0, λ_max, length=N_λ)

t_kgj = [t_fn(λ) for λ in λ_dense]
r_kgj = [r_fn(λ) for λ in λ_dense]
θ_kgj = [θ_fn(λ) for λ in λ_dense]
ϕ_kgj = [ϕ_fn(λ) for λ in λ_dense]

# Build coordinate-time interpolants so we can plot on the same t-axis as yours
itp_r = LinearInterpolation(t_kgj, r_kgj)
itp_θ = LinearInterpolation(t_kgj, θ_kgj)
itp_ϕ = LinearInterpolation(t_kgj, ϕ_kgj)

# Restrict to the t-range covered by both solutions
t_lo = max(t_mine[1],   t_kgj[1])
t_hi = min(t_mine[end], t_kgj[end])
t_common = range(t_lo, t_hi, length=4000)

r_kgj_t = itp_r.(t_common)
θ_kgj_t = itp_θ.(t_common)
ϕ_kgj_t = itp_ϕ.(t_common)

# Interpolate yours onto the same grid too (for residual plots)
itp_r_mine = LinearInterpolation(t_mine, r_mine)
itp_θ_mine = LinearInterpolation(t_mine, θ_mine)
itp_ϕ_mine = LinearInterpolation(t_mine, ϕ_mine)

r_mine_t = itp_r_mine.(t_common)
θ_mine_t = itp_θ_mine.(t_common)
ϕ_mine_t = itp_ϕ_mine.(t_common)

# ---------------------
# 3. Compare! 
# ---------------------
fig = Figure(size = (1000, 900))

ax_r = Axis(fig[1, 1]; xlabel = L"t/M", ylabel = L"r/M", title = "Radial Coordinate")
ax_θ = Axis(fig[2, 1]; xlabel = L"t/M", ylabel = L"r/M", title = "Polar Coordinate")
ax_ϕ = Axis(fig[3, 1]; xlabel = L"t/M", ylabel = L"r/M", title = "Azimuthal Coordinate")

for (ax, y_mine, y_kgj) in zip(
    (ax_r,   ax_θ,   ax_ϕ),
    (r_mine_t, θ_mine_t, ϕ_mine_t),
    (r_kgj_t,  θ_kgj_t,  ϕ_kgj_t))

    lines!(ax, t_common, y_mine; color=:blue, linewidth=2,   label="RefGeodesics.jl")
    lines!(ax, t_common, y_kgj;  color=:red,  linewidth=1.5,
        linestyle=:dash, label="KerrGeodesics.jl")
    axislegend(ax; position=:rt)
end

display(fig)

# -------------------------------------------------------
# 4. Residual plots — the real diagnostic
# -------------------------------------------------------
fig2 = Figure(size=(1000, 700))

ax_dr = Axis(fig2[1,1]; xlabel=L"t/M", ylabel=L"\Delta r",
             title="Residuals: Mine − KerrGeodesics.jl")
ax_dθ = Axis(fig2[2,1]; xlabel=L"t/M", ylabel=L"\Delta\theta")
ax_dϕ = Axis(fig2[3,1]; xlabel=L"t/M", ylabel=L"\Delta\phi")

lines!(ax_dr, t_common, r_mine_t .- r_kgj_t; color=:black, linewidth=1.5)
lines!(ax_dθ, t_common, θ_mine_t .- θ_kgj_t; color=:black, linewidth=1.5)
lines!(ax_dϕ, t_common, ϕ_mine_t .- ϕ_kgj_t; color=:black, linewidth=1.5)

display(fig2)
save("geodesic_residuals.pdf", fig2)

# -------------------------------------------------------
# 6. 3D overlay comparison
# -------------------------------------------------------

# Cartesian coords for YOUR geodesic (full t_mine grid)
x_mine = @. r_mine * sin(θ_mine) * cos(ϕ_mine)
y_mine = @. r_mine * sin(θ_mine) * sin(ϕ_mine)
z_mine = @. r_mine * cos(θ_mine)

# Cartesian coords for KerrGeodesics.jl (dense Mino-time grid, no interpolation needed)
x_kgj = @. r_kgj * sin(θ_kgj) * cos(ϕ_kgj)
y_kgj = @. r_kgj * sin(θ_kgj) * sin(ϕ_kgj)
z_kgj = @. r_kgj * cos(θ_kgj)

# Black hole horizon sphere
rH   = M + sqrt(M^2 - a^2)
θg   = range(0, π,  length=30)
ϕg   = range(0, 2π, length=60)
Xh   = [rH * sin(th) * cos(ph) for th in θg, ph in ϕg]
Yh   = [rH * sin(th) * sin(ph) for th in θg, ph in ϕg]
Zh   = [rH * cos(th)            for th in θg, ph in ϕg]

fig3 = Figure(size=(900, 800), backgroundcolor=:white)

ax3 = Axis3(fig3[1,1];
    aspect    = :data,
    xlabel    = L"x/M",
    ylabel    = L"y/M",
    zlabel    = L"z/M",
    title     = "Geodesic comparison: mine (blue) vs KerrGeodesics.jl (red dashed)",
    azimuth   = 1.2π,
    elevation = 0.2π,
)

# KerrGeodesics.jl — plot first so it sits "behind"
lines!(ax3, x_kgj, y_kgj, z_kgj;
    color     = :red,
    linewidth = 2.5,
    linestyle = :dash,
    label     = "KerrGeodesics.jl"
)

# Yours — on top, colored by r so you can see radial structure
lines!(ax3, x_mine, y_mine, z_mine;
    color      = r_mine,
    colormap   = :cool,
    colorrange = extrema(r_mine),
    linewidth  = 1.5,
    label      = "RefGeodesics.jl"
)

# Black hole
CairoMakie.surface!(ax3, Xh, Yh, Zh;
    color   = :black,
    alpha   = 0.85,
    shading = NoShading
)

axislegend(ax3; position=:rt)

Colorbar(fig3[2,1];
    colormap  = :cool,
    limits    = extrema(r_mine),
    label     = L"r/M",
    vertical  = false,
    width     = Relative(0.5),
    tellwidth = false
)

display(fig3)
save("geodesic_3d_comparison.pdf", fig3)