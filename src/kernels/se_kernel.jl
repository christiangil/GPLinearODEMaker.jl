import GPLinearODEMaker.powers_of_negative_one

"""
    se_kernel(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires 1 hyperparameters.
Likely created using se_kernel_base() as an input.
Use with include("src/kernels/se_kernel.jl").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `["λ"]`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
"""
function se_kernel(
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

    λ = hyperparameters_view[1]

    if dorder_view==[6, 2]
        func = -630*exp((-1/2)*δ^2/λ^2)/λ^8 + 3465*exp((-1/2)*δ^2/λ^2)*δ^2/λ^10 - 2520*exp((-1/2)*δ^2/λ^2)*δ^4/λ^12 + 546*exp((-1/2)*δ^2/λ^2)*δ^6/λ^14 - 42*exp((-1/2)*δ^2/λ^2)*δ^8/λ^16 + exp((-1/2)*δ^2/λ^2)*δ^10/λ^18
    end

    if dorder_view==[5, 2]
        func = -630*exp((-1/2)*δ^2/λ^2)*δ/λ^8 + 945*exp((-1/2)*δ^2/λ^2)*δ^3/λ^10 - 315*exp((-1/2)*δ^2/λ^2)*δ^5/λ^12 + 33*exp((-1/2)*δ^2/λ^2)*δ^7/λ^14 - exp((-1/2)*δ^2/λ^2)*δ^9/λ^16
    end

    if dorder_view==[4, 2]
        func = 60*exp((-1/2)*δ^2/λ^2)/λ^6 - 285*exp((-1/2)*δ^2/λ^2)*δ^2/λ^8 + 165*exp((-1/2)*δ^2/λ^2)*δ^4/λ^10 - 25*exp((-1/2)*δ^2/λ^2)*δ^6/λ^12 + exp((-1/2)*δ^2/λ^2)*δ^8/λ^14
    end

    if dorder_view==[3, 2]
        func = 60*exp((-1/2)*δ^2/λ^2)*δ/λ^6 - 75*exp((-1/2)*δ^2/λ^2)*δ^3/λ^8 + 18*exp((-1/2)*δ^2/λ^2)*δ^5/λ^10 - exp((-1/2)*δ^2/λ^2)*δ^7/λ^12
    end

    if dorder_view==[2, 2]
        func = -6*exp((-1/2)*δ^2/λ^2)/λ^4 + 27*exp((-1/2)*δ^2/λ^2)*δ^2/λ^6 - 12*exp((-1/2)*δ^2/λ^2)*δ^4/λ^8 + exp((-1/2)*δ^2/λ^2)*δ^6/λ^10
    end

    if dorder_view==[1, 2]
        func = -6*exp((-1/2)*δ^2/λ^2)*δ/λ^4 + 7*exp((-1/2)*δ^2/λ^2)*δ^3/λ^6 - exp((-1/2)*δ^2/λ^2)*δ^5/λ^8
    end

    if dorder_view==[0, 2]
        func = -3*exp((-1/2)*δ^2/λ^2)*δ^2/λ^4 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^6
    end

    if dorder_view==[6, 1]
        func = 90*exp((-1/2)*δ^2/λ^2)/λ^7 - 375*exp((-1/2)*δ^2/λ^2)*δ^2/λ^9 + 195*exp((-1/2)*δ^2/λ^2)*δ^4/λ^11 - 27*exp((-1/2)*δ^2/λ^2)*δ^6/λ^13 + exp((-1/2)*δ^2/λ^2)*δ^8/λ^15
    end

    if dorder_view==[5, 1]
        func = 90*exp((-1/2)*δ^2/λ^2)*δ/λ^7 - 95*exp((-1/2)*δ^2/λ^2)*δ^3/λ^9 + 20*exp((-1/2)*δ^2/λ^2)*δ^5/λ^11 - exp((-1/2)*δ^2/λ^2)*δ^7/λ^13
    end

    if dorder_view==[4, 1]
        func = -12*exp((-1/2)*δ^2/λ^2)/λ^5 + 39*exp((-1/2)*δ^2/λ^2)*δ^2/λ^7 - 14*exp((-1/2)*δ^2/λ^2)*δ^4/λ^9 + exp((-1/2)*δ^2/λ^2)*δ^6/λ^11
    end

    if dorder_view==[3, 1]
        func = -12*exp((-1/2)*δ^2/λ^2)*δ/λ^5 + 9*exp((-1/2)*δ^2/λ^2)*δ^3/λ^7 - exp((-1/2)*δ^2/λ^2)*δ^5/λ^9
    end

    if dorder_view==[2, 1]
        func = 2*exp((-1/2)*δ^2/λ^2)/λ^3 - 5*exp((-1/2)*δ^2/λ^2)*δ^2/λ^5 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^7
    end

    if dorder_view==[1, 1]
        func = 2*exp((-1/2)*δ^2/λ^2)*δ/λ^3 - exp((-1/2)*δ^2/λ^2)*δ^3/λ^5
    end

    if dorder_view==[0, 1]
        func = exp((-1/2)*δ^2/λ^2)*δ^2/λ^3
    end

    if dorder_view==[6, 0]
        func = -15*exp((-1/2)*δ^2/λ^2)/λ^6 + 45*exp((-1/2)*δ^2/λ^2)*δ^2/λ^8 - 15*exp((-1/2)*δ^2/λ^2)*δ^4/λ^10 + exp((-1/2)*δ^2/λ^2)*δ^6/λ^12
    end

    if dorder_view==[5, 0]
        func = -15*exp((-1/2)*δ^2/λ^2)*δ/λ^6 + 10*exp((-1/2)*δ^2/λ^2)*δ^3/λ^8 - exp((-1/2)*δ^2/λ^2)*δ^5/λ^10
    end

    if dorder_view==[4, 0]
        func = 3*exp((-1/2)*δ^2/λ^2)/λ^4 - 6*exp((-1/2)*δ^2/λ^2)*δ^2/λ^6 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^8
    end

    if dorder_view==[3, 0]
        func = 3*exp((-1/2)*δ^2/λ^2)*δ/λ^4 - exp((-1/2)*δ^2/λ^2)*δ^3/λ^6
    end

    if dorder_view==[2, 0]
        func = -exp((-1/2)*δ^2/λ^2)/λ^2 + exp((-1/2)*δ^2/λ^2)*δ^2/λ^4
    end

    if dorder_view==[1, 0]
        func = -exp((-1/2)*δ^2/λ^2)*δ/λ^2
    end

    if dorder_view==[0, 0]
        func = exp((-1/2)*δ^2/λ^2)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return se_kernel, 1  # the function handle and the number of kernel hyperparameters
