```

    Inverse problem script to learn ODE model from SXS waveform data 

```

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using DelimitedFiles

# ensure the folder exists
mkpath("./input")

include("DissipativeTraining.jl")
include("Orbit2Waveform.jl")
include("2021EquationFunctions.jl")

gr()

"""
DEFINE INITIAL CONDITIONS
"""
p = 20
e = 0.5

generateDissipativeTraining(p, e, 1, 2.5e3, 500, 1)