
# ======================================================================
# orbit2waveform.jl
# Implementation of Numerical Kludge Orbit to Waveform Mapping 
# ======================================================================

using LinearAlgebra
using CairoMakie
using FFTW

include("Orbit2Waveform.jl")

true_solution = load_solution_txt("input/solution_Kerr_rp11.6044152088094_ra11.6044152088194noise0.0_massRatio1000.0spin_0.9inclination_1.5707963267948966_time400.txt")

# ----------------------------------------------------
# 1. Check Mino Time or Uniform Coordinate Time Grid
# ----------------------------------------------------

function assert_uniform_time(t; rtol = 1e-8)
    dt = t[2] - t[1]
    @assert all(abs.(diff(t) .- dt) .< rtol * abs(dt)) "t must be uniformly sampled in coordinate time"
    return dt
end

# ----------------------------------------------------
# 2. Boyer-Lindquist to Cartesian Coordinates
# ----------------------------------------------------

function BoyerLindquist_to_Cartesian(r, θ, ϕ; a)
    x = @. r * sin(θ) * cos(ϕ)
    y = @. r * sin(θ) * sin(ϕ)
    z = @. r * cos(θ)
    return x, y, z
end 

function Soln_to_Source(soln; a)
    t = soln[1, :]
    r = soln[2, :]
    θ = soln[3, :]
    ϕ = soln[4, :]

    x, y, z = BoyerLindquist_to_Cartesian(r, θ, ϕ; a = a)
    orbit = [x'; y'; z']

    return t, orbit
end

# ----------------------------------------------------
# 3. Compute Quadrupole Tensor
# ----------------------------------------------------

function Calculate_Quadrupole_Tensor(orbit; μ = 1.0)
    N = size(orbit, 2)
    Q = zeros(3, 3, N)

    for k in 1:N 
        x = orbit[:, k]
        r2 = dot(x, x)

        for i in 1:3, j in 1:3 
            δij = (i == j) ? 1.0 : 0.0
            Q[i, j, k] = μ * (x[i] * x[j] - (1/3)*δij*r2)
        end
    end

    return Q
end

# ----------------------------------------------------
# 4. Compute Q̇̇
# ----------------------------------------------------

function Calculate_Q_ddot(Q, dt)
    Qdd = similar(Q)

    for i in 1:3, j in 1:3
        Qdd[i, j, :] .= d2_dt2(vec(Q[i, j, :]), dt)
    end

    return Qdd 
end

# ----------------------------------------------------
# 5. Observer Geometry
# ----------------------------------------------------

function Observer_Basis(θ_obs, ϕ_obs)
    
    # Direction to observer
    n = [
        sin(θ_obs) * cos(ϕ_obs),
        sin(θ_obs) * sin(ϕ_obs),
        cos(θ_obs)
    ]

    # Polarization basis vectors (p,q) on observer sky
    p = [
        cos(θ_obs) * cos(ϕ_obs),
        cos(θ_obs) * sin(ϕ_obs),
        -sin(θ_obs)
    ]

    q = [
        -sin(ϕ_obs),
        cos(ϕ_obs),
        0.0
    ]

    return n, p, q
end

# ----------------------------------------------------
# 6. Compute h₊ and hₓ
# ----------------------------------------------------

function Compute_h₊_hₓ(Qdd; θ_obs = 0.0, ϕ_obs = 0.0, D = 1.0)
    n, p, q = Observer_Basis(θ_obs, ϕ_obs)
    N = size(Qdd, 3)
    
    h₊ = zeros(N)
    hₓ = zeros(N)

    Pij = I - n * n' 

    for k in 1:N

        H = (2/D) .* Qdd[:, :, k]
        HTT = Pij * H * Pij - 0.5 * tr(Pij * H) * Pij

        h₊[k] = sum(
            (p[i] * p[j] - q[i] * q[j]) * HTT[i, j]
            for i in 1:3, j in 1:3
        )

        hₓ[k] = sum(
            (p[i] * q[j] + q[i] * p[j]) * HTT[i, j]
            for i in 1:3, j in 1:3
        )
    end

    return h₊, hₓ
end

# ----------------------------------------------------
# 7. Calculate Orbit to Waveform
# ----------------------------------------------------

function nk_quadrupole_waveform(soln; a, μ = 1.0, θ_obs = 0.0, ϕ_obs = 0.0, D = 1.0)
    
    t, orbit = Soln_to_Source(soln; a = a)
    dt = assert_uniform_time(t)
    Q = Calculate_Quadrupole_Tensor(orbit; μ = μ)
    Q̇̇ = Calculate_Q_ddot(Q, dt)
    h₊, hₓ = Compute_h₊_hₓ(Q̇̇; θ_obs = θ_obs, ϕ_obs = ϕ_obs, D = D)

    return t, h₊, hₓ, Q, Q̇̇
end 

t, h₊, hₓ, Q, Qdd = nk_quadrupole_waveform(true_solution; a = 0.9, μ = 1.0, θ_obs = 1.5707963267948966, ϕ_obs = 0.0, D = 1.0)

t_plot = vec(t)
h₊_plot = vec(h₊)
hₓ_plot = vec(hₓ)

fig_h = Figure(size = (900, 450))
ax_h = CairoMakie.Axis(fig_h[1, 1];
    xlabel = L"t/M",
    ylabel = L"h(t)",
    title = L"h(t)"
)

lines!(ax_h, t, h₊; color = :blue, linewidth = 2, label = L"h_+")
display(fig_h)

# fig_hx = Figure(size = (900, 450))
# ax_hx = CairoMakie.Axis(fig_hx[1, 1];
#     xlabel = L"t/M",
#     ylabel = L"h_\times(t)",
#     title = L"h_\times(t)"
# )

lines!(ax_h, t, hₓ; color = :red, linewidth = 2, label = L"h_\times")

axislegend(ax_h; position = :rt)
display(fig_h)

function tensor_sanity_checks(Q, Qdd)
    N = size(Q, 3)

    trQ   = [Q[1,1,k] + Q[2,2,k] + Q[3,3,k] for k in 1:N]
    trQdd = [Qdd[1,1,k] + Qdd[2,2,k] + Qdd[3,3,k] for k in 1:N]

    symQ   = maximum(abs.(Q .- permutedims(Q, (2,1,3))))
    symQdd = maximum(abs.(Qdd .- permutedims(Qdd, (2,1,3))))

    println("max |tr(Q)|   = ", maximum(abs.(trQ)))
    println("max |tr(Qdd)| = ", maximum(abs.(trQdd)))
    println("max |Q-Qᵀ|    = ", symQ)
    println("max |Qdd-Qddᵀ|= ", symQdd)
end

tensor_sanity_checks(Q, Qdd)

function faceon_projection_check(h₊, hₓ, Qdd; D=1.0)
    h₊_manual = (2/D) .* (vec(Qdd[1,1,:]) .- vec(Qdd[2,2,:]))
    hₓ_manual = (4/D) .* vec(Qdd[1,2,:])

    println("max h+ projection error = ", maximum(abs.(h₊ .- h₊_manual)))
    println("max hx projection error = ", maximum(abs.(hₓ .- hₓ_manual)))
end

faceon_projection_check(h₊, hₓ, Qdd; D = 1.0)

_, hp1, hx1, _, _ = nk_quadrupole_waveform(true_solution; a=0.9, μ=1.0, θ_obs=π/4, ϕ_obs=π/6, D=1.0)
_, hp2, hx2, _, _ = nk_quadrupole_waveform(true_solution; a=0.9, μ=1.0, θ_obs=π/4, ϕ_obs=π/6, D=2.0)

println("D scaling h+ error = ", maximum(abs.(hp2 .- hp1 ./ 2)))
println("D scaling hx error = ", maximum(abs.(hx2 .- hx1 ./ 2)))

_, hp1, hx1, _, _ = nk_quadrupole_waveform(true_solution; a=0.9, μ=1.0, θ_obs=π/4, ϕ_obs=π/6, D=1.0)
_, hp2, hx2, _, _ = nk_quadrupole_waveform(true_solution; a=0.9, μ=2.0, θ_obs=π/4, ϕ_obs=π/6, D=1.0)

println("μ scaling h+ error = ", maximum(abs.(hp2 .- 2hp1)))
println("μ scaling hx error = ", maximum(abs.(hx2 .- 2hx1)))

function toy_circular_solution(; R=10.0, Ω=0.03, T=1000.0, N=5000)
    t = collect(range(0, T, length=N))
    r = fill(R, N)
    θ = fill(π/2, N)
    ϕ = Ω .* t

    # dummy momenta
    return [t'; r'; θ'; ϕ'; zeros(N)'; zeros(N)'; zeros(N)'; zeros(N)']
end

toy = toy_circular_solution()

t, hp, hx, Q, Qdd = nk_quadrupole_waveform(
    toy; a=0.0, μ=1.0, θ_obs=0.0, ϕ_obs=0.0, D=1.0
)

fig = Figure(size=(900,450))
ax = CairoMakie.Axis(fig[1,1], title="Toy circular equatorial face-on waveform")
lines!(ax, t, hp; label=L"h_+", color=:blue)
lines!(ax, t, hx; label=L"h_\times", color=:red)
axislegend(ax)
display(fig)

function dominant_frequency(t, h)
    dt = t[2] - t[1]
    N = length(h)
    H = abs.(fft(h .- mean(h)))
    freqs = (0:N-1) ./ (N*dt)

    half = 2:div(N,2) # skip zero frequency
    imax = half[argmax(H[half])]
    return freqs[imax]
end

f_dom = dominant_frequency(t, hp)
println("dominant angular freq ≈ ", 2π*f_dom)
println("expected angular freq = ", 2*0.03)

for θobs in [0.0, π/4, π/2]
    _, hp, hx, _, _ = nk_quadrupole_waveform(
        toy; a=0.0, μ=1.0, θ_obs=θobs, ϕ_obs=0.0, D=1.0
    )
    println("θobs = $θobs")
    println("amp h+ ≈ ", maximum(abs.(hp)))
    println("amp hx ≈ ", maximum(abs.(hx)))
end

fig = Figure(size=(900,600))
ax1 = CairoMakie.Axis(fig[1,1], ylabel=L"r(t)", title="Periapsis vs waveform")
ax2 = CairoMakie.Axis(fig[2,1], xlabel=L"t/M", ylabel=L"|h|")

hamp = sqrt.(h₊.^2 .+ hₓ.^2)

lines!(ax1, t, true_solution[2,:]; color=:black)
lines!(ax2, t, hamp; color=:blue)

display(fig)

t_kerr, h₊_kerr, hₓ_kerr, Q_kerr, Qdd_kerr = nk_quadrupole_waveform(
    true_solution;
    a = 0.9,
    μ = 1.0,
    θ_obs = 0.0,
    ϕ_obs = 0.0,
    D = 1.0
)

r_kerr = vec(true_solution[2, :])
hamp_kerr = sqrt.(h₊_kerr.^2 .+ hₓ_kerr.^2)

fig = Figure(size=(900,600))

ax1 = CairoMakie.Axis(fig[1,1], ylabel=L"r(t)", title="Periapsis vs waveform")
ax2 = CairoMakie.Axis(fig[2,1], xlabel=L"t/M", ylabel=L"|h|")

lines!(ax1, t_kerr, r_kerr; color=:black)
lines!(ax2, t_kerr, hamp_kerr; color=:blue)

display(fig)