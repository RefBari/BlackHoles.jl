using Documenter
using BlackHoles

makedocs(
    sitename  = "BlackHoles.jl",
    modules   = [BlackHoles],
    format    = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages     = ["Home" => "index.md"],
    strict    = false,      # <-- do not fail on warnings in CI
    checkdocs = :none,      # <-- do not require every export to have a docstring
)

deploydocs(
    repo      = "github.com/RefBari/BlackHoles.jl.git",
    devbranch = "main",
)
