push!(LOAD_PATH,"../src/")
using Documenter, GPLinearODEMaker

makedocs(
    modules=[GPLinearODEMaker],
    sitename="GPLinearODEMaker.jl",
    authors = "Christian Gilbertson",
    pages = ["Home" => "index.md"])
