```

    Inverse problem script to learn ODE model from SXS waveform data

```
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using OrdinaryDiffEq
using Optim
using LineSearches
using DiffEqFlux
using DiffEqSensitivity
using Plots
using DataFrames
using CSV
gr()

include("utils.jl")
include("models.jl")

## Define the experiment parameters
u0 = Float32[pi, 0.0, 12.37, 1.0f-4] # χ₀, ϕ₀, p₀, e₀
tspan = (5.20913f3, 6.78f3)
datasize = 1000
tsteps = range(tspan[1], tspan[2], length = datasize)
dt_data = tsteps[2] - tsteps[1]
dt = 10.0
num_optimization_increments = 100
mass_ratio = 1.0
model_params = [mass_ratio] # Just the mass ratio, at least for now
mass1 = mass_ratio/(1.0+mass_ratio)
mass2 = 1.0/(1.0+mass_ratio)

# Gather waveform data
x, y = file2trajectory(tsteps,"./input/trajectoryA.txt")
waveform_real = file2waveform(tsteps,"./input/waveform_real.txt")
x2, y2 = file2trajectory(tsteps,"./input/trajectoryB.txt")

plt = plot(x,y)
