cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using LaTeXStrings
using Measures
using SymbolicRegression
using LinearAlgebra
using OrdinaryDiffEq
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using Random
using SciMLBase
using SciMLSensitivity
using OptimizationOptimJL
using ForwardDiff
using ComponentArrays
using Optim
using Optimization
using OptimizationOptimisers
using Plots
using Lux
using Zygote

include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/GaussianModel.jl")
include("/Users/rbari/Downloads/common_denominator-2020_scimlforbbhs-e19e6808aad2/paper/zenodo/GWaveInversion/WaveformMan.jl")

mass_ratio = 1
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = 1.0/(1.0+mass_ratio)
mass2 = mass_ratio/(1.0+mass_ratio)

tspan = (0, 2e3)
datasize = 100
tsteps = range(tspan[1], tspan[2], length = datasize+1) 
dt_data = tsteps[2] - tsteps[1]
dt = 1
num_optimization_increments = 100

function ODE_model_Newtonian(du, u, p, t)
    du = GENERIC_Newtonian(du, u, model_params, t)
    return du
end

function ODE_model_Schwarzschild(du, u, p, t)
    du = GENERIC_Schwarzschild(du, u, model_params, t)
    return du
end

p = 20
e = 0.1

R = p / (1-e)

E0_newtonian, L0_newtonian = pe_2_EL_newtonian(p, e)
E0_schwarzschild, L0_schwarzschild = pe_2_EL(p, e)[2:3]

u0_newtonian = [0.0, R, pi/2, 0.0, -E0_newtonian, 0.0, 0.0, L0_newtonian, 0.0]
u0_schwarzschild = [0.0, R, pi/2, 0.0, -E0_schwarzschild, 0.0, 0.0, L0_schwarzschild, 0.0]

prob_schwarzschild = ODEProblem(ODE_model_Schwarzschild, u0_schwarzschild, tspan)
soln_schwarzschild = Array(solve(prob_schwarzschild, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
orbit_schwarzschild = soln2orbit(soln_schwarzschild)
blackHole_r1_sch, blackHole_r2_sch = one2two(orbit_schwarzschild, mass1, mass2)
h_plus_true_sch, h_cross_true_sch = h_22_strain_two_body(dt_data, blackHole_r1_sch, mass1, blackHole_r2_sch, mass2)
plot(orbit_schwarzschild[1,:], orbit_schwarzschild[2,:], linewidth = 2)
plot(h_plus_true_sch[1:13])

prob_newton = ODEProblem(ODE_model_Newtonian, u0_newtonian, tspan)
soln_newton = Array(solve(prob_newton, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
orbit_newton = soln2orbit(soln_newton)
blackHole_r1_newt, blackHole_r2_newt = one2two(orbit_newton, mass1, mass2)
h_plus_true_newt, h_cross_true_newt = h_22_strain_two_body(dt_data, blackHole_r1_newt, mass1, blackHole_r2_newt, mass2)
plot!(orbit_newton[1,1:18], orbit_newton[2,1:18], linewidth = 2)
plot!(h_plus_true_newt[1:13])