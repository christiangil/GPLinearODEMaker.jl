"""
white_kernel function. Requires 0 hyperparameters.
Use with include("src/kernels/white_kernel.jl").
hyperparameters == []
"""
function white_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    if all(dorder .== 0)
        return Int(δ == 0)
    end

    return 0

end


return white_kernel, 0  # the function handle and the number of kernel hyperparameters
