using ScatteringOptics, VLBISkyModels, CairoMakie, Statistics, LinearAlgebra

# --- Choose a scattering model (J18-like). You can tune α, rin_km, anisotropy, PA later.
sm = ScatteringModel()  # defaults are reasonable; see ?ScatteringModel to set α, r_in, etc.

# Frequencies to test (Hz)
νs = [86e9, 120e9, 138e9, 150e9, 230e9]

# Baselines (m/λ units are handled inside visibility_point; we'll pass u in wavelengths)
uλ = LinRange(0, 30e9, 2000)   # 0..30 Gλ along the u-axis
vλ = LinRange(0, 30e9, 2000)   # same for v-axis

# Helper: estimate Gaussian FWHM from visibility slope vs b^2.
# For a circular Gaussian with FWHM θ, |V| = exp[-π^2 b^2 θ^2 / (4 ln 2)].
# => ln|V| = - (π^2 θ^2 / (4 ln 2)) * b^2  + const  => slope = -C θ^2, C = π^2/(4 ln 2)
const C = (π^2) / (4 * log(2))

function fwhm_from_vis(uλ_vals; ν, along=:u)
    # sample along one axis (major/minor will differ if anisotropic)
    pts = if along == :u
        [(U=u, V=0.0, Fr=ν) for u in uλ_vals]
    else
        [(U=0.0, V=v, Fr=ν) for v in uλ_vals]
    end
    km = kernelmodel(sm, νref=ν)
    vis = abs.([visibility_point(km, p) for p in pts])

    # use only the decaying part (exclude DC and near-zero baselines)
    mask = (uλ_vals .>= 1e8) .& (vis .> 0)  # ≥0.1 Gλ; adjust if needed
    b2 = (uλ_vals[mask].^2)
    y  = log.(vis[mask])

    # linear regression slope
    slope = cov(b2, y) / var(b2)
    θ = sqrt(-slope / C)               # in radians
    θ_mas = θ * (180/π) * 3.6e6 * 1e3  # radians -> mas
    return θ_mas, uλ_vals, vis
end

fwhm_major = Float64[]; fwhm_minor = Float64[]; freqs_GHz = Float64[]

for ν in νs
    θu, ugrid, visu = fwhm_from_vis(uλ; ν=ν, along=:u)
    θv, vgrid, visv = fwhm_from_vis(vλ; ν=ν, along=:v)
    push!(fwhm_major, θu); push!(fwhm_minor, θv); push!(freqs_GHz, ν/1e9)
end

# --- Plot FWHM vs frequency
f1 = Figure(resolution=(800,420))
ax1 = Axis(f1[1,1], xlabel="Frequency (GHz)", ylabel="FWHM (mas)", yscale=log10, xscale=log10)
scatterlines!(ax1, freqs_GHz, fwhm_major, label="Major-axis", markersize=8)
scatterlines!(ax1, freqs_GHz, fwhm_minor, label="Minor-axis", markersize=8)
axislegend(ax1)
f1
