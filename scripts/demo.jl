using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BlackHoles

println(BlackHoles.addition(1, 2))

println(BlackHoles.pe_2_EL(1.0, 2.0))
