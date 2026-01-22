```

    Inverse problem script to recover geodesic equations from waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
# using OrdinaryDiffEq
# using DiffEqFlux
using Optim
using LineSearches
using DataFrames
using CSV
using Plots
using DifferentialEquations
using LaTeXStrings
gr()

include("utils.jl")
include("models.jl")

## Define the experimental parameters
mass_ratio = 1.0
u0 = Float64[0, 0.0]
datasize = 250
tspan = (0.0f0, 6.0f4)
tsteps = range(tspan[1], tspan[2], length = datasize)
dt_data = tsteps[2] - tsteps[1]
dt = 100.0
model_params = [100.0, 1.0, 0.5] # p, M, e

# Generate waveform data
prob = ODEProblem(RelativisticOrbitModel, u0, tspan, model_params)
soln = Array(solve(prob, RK4(), saveat = tsteps, dt = dt, adaptive=false))
waveform = compute_waveform(dt_data, soln, mass_ratio, model_params)[1]

plt = plot!(tsteps, waveform,
           markershape=:circle, markeralpha = 0.25,
           linewidth = 2, alpha = 0.5,
           label=L"GW using $\dot{\phi},\dot{\chi}$ (Orbital Parameters)")