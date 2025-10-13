using Documenter
using BlackHoles

makedocs(
    sitename  = "BlackHoles.jl",
    modules   = [BlackHoles],
    format    = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages     = ["Home" => "index.md"],
    checkdocs = :none,                 # don’t enforce docstrings right now
    warnonly  = [:missing_docs],       # treat missing docs as warnings, not errors
)

deploydocs(
    repo      = "github.com/RefBari/BlackHoles.jl.git",
    devbranch = "main",
)
