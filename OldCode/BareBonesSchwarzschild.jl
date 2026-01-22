```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Printf
using DelimitedFiles

# ensure the folder exists
mkpath("./input")

# Clean imports - no duplicates
using LaTeXStrings
using Measures
using LinearAlgebra
using OrdinaryDiffEq
using Optim
using LineSearches
using DataFrames
using PrettyTables
using CSV
using Plots
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
using ComponentArrays
using Optimization
using OptimizationOptimisers
using Lux

gr()

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/SchwarzschildModelsTraining.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")


"""
DEFINE INITIAL CONDITIONS
"""
mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (0, 2.00001e3)
datasize = 100
tsteps = range(tspan[1], tspan[2], length = datasize+1) 
dt_data = tsteps[2] - tsteps[1]
dt = 1
num_optimization_increments = 100

p = 18
e = 0.1

R = p / (1 - e)
pᵣ0 = 0

# E0, L0 = circular_pt_L(R)
# E0, L0 = eccentric_pt_L(p, e)
E0, L0 = pe_2_EL(p, e)[2:3] # Schwarzschild ICs
u0 = [0.0, R, pi/2, 0.0, -E0, 0.0, 0.0, L0, 0.0]

"""
DEFINE ODE MODEL FOR BINARY BLACK HOLE SYSTEM!
"""
function ODE_model_dual(du, u, p, t)
    du = GENERIC(du, u, model_params, t)
    return du
end

"""
SET UP ODE MODEL + SOLVE IT + CONVERT TO WAVEFORM
"""
prob = ODEProblem(ODE_model_dual, u0, tspan)
soln = Array(solve(prob, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
orbit = soln2orbit(soln)
blackHole_r1, blackHole_r2 = one2two(orbit, mass1, mass2)
h_plus_true, h_cross_true = h_22_strain_two_body(dt_data, blackHole_r1, mass1, blackHole_r2, mass2)
# r_min, r_max = p/(1+e), p/(1-e)
# r = range(r_min, r_max)

# g_rr_terms = []
# g_ϕϕ_terms = []

# r_schwarzschild = soln[2,:]
# pₜ_schwarzschild = soln[5,:]
# pᵣ_schwarzschild = soln[6,:]
# ṙ_schwarzschild = zeros(length(r_schwarzschild))

# for i in 2:length(r_schwarzschild)-1
#     ṙ_schwarzschild[i] = (r_schwarzschild[i+1] - r_schwarzschild[i-1]) / (2*(tsteps[i+1] - tsteps[i]))
# end

# r_schwarzschild_plot = plot(r_schwarzschild, label = L"r")
# pt_schwarzschild_plot = plot(pₜ_schwarzschild, label = L"p_t")
# pᵣ_schwarzschild_plot = plot(pᵣ_schwarzschild, label = L"p_r")
# ṙ_schwarzschild_plot = plot(ṙ_schwarzschild, label = L"\dot{r}")
# grr_schwarzschild_plot = plot(g_rr.(r_schwarzschild), label = L"g^{rr}")
# gtt_schwarzschild_plot = plot(g_tt.(r_schwarzschild), label = L"g_{tt}")
# ṙ_alternative_schwarzschild = ( g_rr.(r_schwarzschild) .* pᵣ_schwarzschild ) ./ ( g_tt.(r_schwarzschild) .* pₜ_schwarzschild )
# plot!(ṙ_alternative_schwarzschild, label = "Alternative ṙ")

# for i in range(r_min, r_max)
#     push!(g_rr_terms, g_rr(i))
#     push!(g_ϕϕ_terms, g_ϕϕ(i))
# end

# grr_plot = scatter(r, g_rr_terms, xlabel = L"r", ylabel = L"g^{rr}", linewidth = 2, title = L"g^{rr}", legend = false)
# plot!(r, g_rr_terms, xlabel = L"r", ylabel = L"g^{rr}", linewidth = 2)
# gϕϕ_plot = scatter(r, g_ϕϕ_terms, xlabel = L"r", ylabel = L"g^{ϕϕ}", linewidth = 2, title = L"g^{ϕϕ}", legend = false)
# plot!(r, g_ϕϕ_terms, xlabel = L"r", ylabel = L"g^{ϕϕ}", linewidth = 2)

# metric_comps = [r g_rr_terms g_ϕϕ_terms ;]

# pretty_table(
#     metric_comps;
#     header            = ["r", "g^rr", "g^ϕϕ"],
# )
                
plot(blackHole_r1[1,:], blackHole_r1[2,:], aspect_ratio=:equal, linewidth = 2, label = "Black Hole 1")
plot!(blackHole_r2[1,:], blackHole_r2[2,:], aspect_ratio=:equal, linewidth = 2, legend=:topleft, label = "Black Hole 2")
plot(h_plus_true, linewidth = 2, legend=:topleft, label = L"h_+")
plot!(h_cross_true, linewidth = 2, legend=:topleft, label = L"$h_\times$")

x1, y1 = blackHole_r1[1, :], blackHole_r1[2, :]
x2, y2 = blackHole_r2[1, :], blackHole_r2[2, :]
hplus  = h_plus_true
hcross = h_cross_true

"""
EXPORT TRAINING DATA
"""
t_vec = collect(tsteps)
t0 = t_vec[1]
t_out = t_vec .- t0  

function write_trajectory_txt(path::AbstractString, t::AbstractVector,
                              x::AbstractVector, y::AbstractVector)
    @assert length(t) == length(x) == length(y)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e %.18e\n", t[i], x[i], y[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

function write_waveform_txt(path::AbstractString, t::AbstractVector, h::AbstractVector)
    @assert length(t) == length(h)
    open(path, "w") do io
        for i in eachindex(t)
            @printf(io, "%.18e %.18e\n", t[i], h[i])
        end
    end
    println("Wrote $(length(t)) rows → $path")
end

write_trajectory_txt("./input/trajectoryA_Schwarzschild_p18_e0p1.txt", t_out, x1, y1)
write_trajectory_txt("./input/trajectoryB_Schwarzschild_p18_e0p1.txt", t_out, x2, y2)
write_waveform_txt("./input/waveform_real_Schwarzschild_p18_e0p1.txt", t_out, hplus)
write_waveform_txt("./input/waveform_imag_Schwarzschild_p18_e0p1.txt", t_out, hcross)