using Documenter
using BlackHoles

makedocs(
    sitename = "BlackHoles.jl",
    modules  = [BlackHoles],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo      = "github.com/RefBari/BlackHoles.jl.git",
    devbranch = "main",          # change to "master" if that’s your default
)
