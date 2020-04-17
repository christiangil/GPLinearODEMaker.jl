"""
zero kernel function
"""
function zero_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    return 0

end


return zero_kernel, 0  # the function handle and the number of kernel hyperparameters
