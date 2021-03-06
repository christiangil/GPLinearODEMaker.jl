# push!(LOAD_PATH,"../src/")
using Documenter, GPLinearODEMaker

DocMeta.setdocmeta!(GPLinearODEMaker, :DocTestSetup, :(using GPLinearODEMaker); recursive=true)

makedocs(
    modules = [GPLinearODEMaker],
    sitename = "GPLinearODEMaker.jl",
    authors = "Christian Gilbertson",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "gettingstarted.md",
        "User's Guide" => [
            "Kernels" => [
                "Kernel functions" => "kernel.md",
                "Adding new kernels" => "kernel_creation.md",
                ],
            "GPLinearODE struct" => "glo.md",
            "GLOM Functionality" => "nlogl.md",
            "Prior functions" => "priors.md",
        ],
        "LICENSE.md",
        hide("Indices" => "indices.md"),
        hide("Diagnostic functions" => "diagnostic.md"),
        hide("Utility functions" => "utility.md"),
    ]
)

deploydocs(
    repo = "github.com/christiangil/GPLinearODEMaker.jl.git",
    deploy_config = Documenter.GitHubActions(),
)
