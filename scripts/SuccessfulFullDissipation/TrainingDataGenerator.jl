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
p = 10
e = 0.4

generate_training_data(p, e, 1, 2e3, 500, 1, "schwarzschild")
generate_training_data(p, e, 1, 2e3, 500, 1, "synthetic_grr")
generate_training_data(p, e, 1, 2e3, 500, 1, "NN")
generate_training_data(p, e, 1, 2e3, 500, 1, "dissipation")