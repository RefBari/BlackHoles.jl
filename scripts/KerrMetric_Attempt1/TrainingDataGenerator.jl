```

Generate Training Data for Conservative Dynamics

```

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using DelimitedFiles

# ensure the folder exists
mkpath("./input")

include("TrainingDataFunctions.jl")
include("Orbit2Waveform.jl")
include("2021EquationFunctions.jl")

gr()

"""
DEFINE INITIAL CONDITIONS
"""
# p, e = 14, 0.2

# generate_training_data(p, e, 1, 2e3, 500, 1, "schwarzschild", 0.02)
# generate_training_data(p, e, 1, 2e3, 500, 1, "synthetic_grr")
# generate_training_data(p, e, 1, 2e3, 500, 1, "NN")
# generate_training_data(p, e, 1, 2e3, 500, 1, "dissipation")

r_min = 11.6044152088094
r_max = 11.6044152088194
θ_min = π/2
a = 0.9
mass_ratio = 1e3
max_time = 400
num_datapoints = 4000

generate_training_data(r_min, r_max, θ_min, a, mass_ratio, max_time, num_datapoints, 1, "kerr", 0.0)