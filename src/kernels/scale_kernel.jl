import GPLinearODEMaker.powers_of_negative_one

"""
    scale_kernel(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires 1 hyperparameters.
Likely created using scale_kernel_base() as an input.
Use with include("src/kernels/scale_kernel.jl").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `["σ"]`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
"""
function scale_kernel(
    hyperparameters::AbstractVector{<:Real},
    δ::Real,
    dorder::AbstractVector{<:Integer})

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    dorder_len = 3
    @assert length(dorder)==dorder_len "dorder is the wrong length"
    @assert maximum(dorder) < 5 "No more than two derivatives for each time or hyperparameter can be calculated"
    @assert minimum(dorder) >= 0 "No integrals"

    dorder2 = dorder[2]
    dorder[2] += dorder[1]

    dorder_view = view(dorder, 2:dorder_len)
    
    σ = hyperparameters[1]

    if dorder_view==[6, 2]
        func = 0
    end

    if dorder_view==[5, 2]
        func = 0
    end

    if dorder_view==[4, 2]
        func = 0
    end

    if dorder_view==[3, 2]
        func = 0
    end

    if dorder_view==[2, 2]
        func = 0
    end

    if dorder_view==[1, 2]
        func = 0
    end

    if dorder_view==[0, 2]
        func = 2
    end

    if dorder_view==[6, 1]
        func = 0
    end

    if dorder_view==[5, 1]
        func = 0
    end

    if dorder_view==[4, 1]
        func = 0
    end

    if dorder_view==[3, 1]
        func = 0
    end

    if dorder_view==[2, 1]
        func = 0
    end

    if dorder_view==[1, 1]
        func = 0
    end

    if dorder_view==[0, 1]
        func = 2*σ
    end

    if dorder_view==[6, 0]
        func = 0
    end

    if dorder_view==[5, 0]
        func = 0
    end

    if dorder_view==[4, 0]
        func = 0
    end

    if dorder_view==[3, 0]
        func = 0
    end

    if dorder_view==[2, 0]
        func = 0
    end

    if dorder_view==[1, 0]
        func = 0
    end

    if dorder_view==[0, 0]
        func = σ^2
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return scale_kernel, 1  # the function handle and the number of kernel hyperparameters
