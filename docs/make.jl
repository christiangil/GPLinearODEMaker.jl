push!(LOAD_PATH,"../src/")
using Documenter, GPLinearODEMaker

makedocs(
    modules=[GPLinearODEMaker],
    sitename="GPLinearODEMaker.jl",
    authors = "Christian Gilbertson")
    # pages = ["Home" => "index.md"])

deploydocs(repo = "github.com/christiangil/GPLinearODEMaker.jl.git",
           target = "build",
           push_preview = true)
