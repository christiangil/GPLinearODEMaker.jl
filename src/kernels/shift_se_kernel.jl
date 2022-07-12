se_kernel, se_n_hyper = include("se_kernel.jl")
# scale_kernel, scale_n_hyper = include("scale_kernel.jl")

"""
Custom function created by combining kernels from kernel_coder().
hyperparameters == ["λ, shift"]
"""
function shift_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real,
    dorder::Vector{<:Integer};
    outputs::Vector{<:Integer}=[1,1])

    @assert length(hyperparameters)==2 "hyperparameters is the wrong length"
    dorder_len = 4
    @assert length(dorder)==dorder_len "dorder is the wrong length"
    @assert maximum(dorder) < 3 "No more than two derivatives for each time or hyperparameter can be calculated"
    @assert minimum(dorder) >= 0
    @assert maximum(outputs) < 3
    @assert minimum(outputs) > 0

    λ = hyperparameters[1]
    shift = hyperparameters[2]

    # dλ = dorder[3]
    # dshift = dorder[4]

    if dorder[4]>0 && variance; return 0 end

    if variance; return se_kernel([λ], δ, dorder[1:3]) end

    if outputs==[2,1]
        return se_kernel([λ, -shift], δ, dorder; shift_ind=2)
    end

    if outputs==[1,2]
        return se_kernel([λ, shift], δ, dorder; shift_ind=2)
    end


end

return shift_kernel, se_n_hyper+1  # the function handle and the number of kernel hyperparameters

# function shift_kernel(
#     hyperparameters::Vector{<:Real},
#     δ::Real,
#     dorder::Vector{<:Integer};
#     outputs::Vector{<:Integer}=[1,1])
#
#     @assert length(hyperparameters)==4 "hyperparameters is the wrong length"
#     dorder_len = 6
#     @assert length(dorder)==dorder_len "dorder is the wrong length"
#     @assert maximum(dorder) < 3 "No more than two derivatives for each time or hyperparameter can be calculated"
#     @assert minimum(dorder) >= 0
#     @assert maximum(outputs) < 3
#     @assert minimum(outputs) > 0
#
#     λ = hyperparameters[1]
#     shift = hyperparameters[2]
#     σ1 = hyperparameters[3]
#     σ2 = hyperparameters[4]
#
#     # dλ = dorder[3]
#     # dshift = dorder[4]
#     dσ1 = dorder[5]
#     dσ2 = dorder[6]
#
#     if outputs==[1,1]
#         return se_kernel([λ], δ, dorder[1:3]) + scale_kernel([σ1], δ, append!(dorder[1:2],[dσ2])) * Int(δ == 0)
#     end
#
#     if outputs in [[1,2],[2,1]]
#         return se_kernel([λ, -shift], δ, dorder[1:4]; shift_ind=2)
#     end
#
#     if outputs==[2,2]
#         return se_kernel([λ], δ, dorder[1:3]) + scale_kernel([σ2], δ, append!(dorder[1:2],[dσ2])) * Int(δ == 0)
#     end
#
# end
