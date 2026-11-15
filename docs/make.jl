using Documenter
using LocalInformation

DocMeta.setdocmeta!(LocalInformation, :DocTestSetup, :(using LocalInformation); recursive = true)

makedocs(;
    modules = [LocalInformation],
    sitename = "LocalInformation.jl",
    authors = "Denizhan Pak",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://denizhanpak.github.io/LocalInformation.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Estimators" => "estimators.md",
        "GPU" => "gpu.md",
        "Partial information decomposition" => "pid.md",
        "Integrated information decomposition" => "phiid.md",
        "API" => "api.md",
        "Internals" => "internals.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(; repo = "github.com/denizhanpak/LocalInformation.jl", devbranch = "main", push_preview = true)
