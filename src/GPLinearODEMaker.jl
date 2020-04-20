module GPLinearODEMaker

using LinearAlgebra
using SpecialFunctions
using Random
using Distributed
using SharedArrays
using SymEngine
using IterativeSolvers
using Statistics

# using Pkg
# Pkg.activate(".")
# Pkg.add("LinearAlgebra")
# Pkg.add("SpecialFunctions")
# Pkg.add("Random")
# Pkg.add("Distributed")
# Pkg.add("SharedArrays")
# Pkg.add("SymEngine")
# Pkg.pin(PackageSpec(name="SymEngine", version="0.6.0"))
# Pkg.add("IterativeSolvers")
# Pkg.add("Documenter")
# Pkg.add("Statistics")

import Base.ndims

include("general_functions.jl")
include("problem_definition_functions.jl")
include("gp_functions.jl")
include("prior_functions.jl")
include("kernel_base_functions.jl")
include("kernel_creation_functions.jl")
include("diagnostic_functions.jl")

end # module
