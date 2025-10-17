using Documenter
using BlackHoles

makedocs(
    sitename  = "BlackHoles.jl",
    modules   = [BlackHoles],
    format    = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Conservative Dynamics" => [
            "Physics of Schwarzschild Metric"    => "guide/schwarzschild-metric.md",
            "Simulating Orbits in Schwarzschild Metric"  => "guide/orbits.md",
            "Quadrupole Approximation: Orbits to Gravitational Wave"        => "guide/waveforms.md",
            "Adding Higher Order Modes"        => "guide/higher_modes.md",
            "Adding a Neural Network: Waves to Orbits"        => "guide/inverse_problem.md"
        ],
        "Dissipative Dynamics" => [
            "Theory: GENERIC Formalism"    => "guide/GENERIC.md",
            "Simulating Orbits with GENERIC"  => "guide/dissipative_orbits.md",
            "Dissipative Case: Waves to Orbits" => "guide/full_inverse_problem.md"
        ],
        "API"  => "api.md",
    ],
    checkdocs = :none,
    warnonly  = [:missing_docs],
)


deploydocs(
    repo      = "github.com/RefBari/BlackHoles.jl.git",
    devbranch = "main",
)
