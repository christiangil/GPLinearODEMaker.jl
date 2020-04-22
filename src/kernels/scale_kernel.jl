import GPLinearODEMaker.powers_of_negative_one

"""
scale_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using scale_kernel_base() as an input.
Use with include("src/kernels/scale_kernel.jl").
hyperparameters == ["σ"]
"""
function scale_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real,
    dorder::Vector{<:Integer};
    shift_ind::Integer=0)

    include_shift = shift_ind!=0
    @assert length(hyperparameters)==1+Int(include_shift) "hyperparameters is the wrong length"
    dorder_len = 3 + Int(include_shift)
    @assert length(dorder)==dorder_len "dorder is the wrong length"
    @assert maximum(dorder) < 3 "No more than two derivatives for each time or hyperparameter can be calculated"

    dorder2 = dorder[2]
    dorder[2] += dorder[1]

    if include_shift
        dorder[2] += dorder[shift_ind+2]
        δ += hyperparameters[shift_ind]
        hyperparameters_view = view(hyperparameters, 1:2 .!= shift_ind)
        dorder_view = view(dorder, 2:dorder_len)
        dorder_view = view(dorder_view, 1:(dorder_len-1) .!= (shift_ind+1))
    else
        hyperparameters_view = hyperparameters
        dorder_view = view(dorder, 2:dorder_len)
    end

    σ = hyperparameters_view[1]

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
