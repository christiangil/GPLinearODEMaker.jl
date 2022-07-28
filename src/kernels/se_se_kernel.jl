import GPLinearODEMaker.powers_of_negative_one

"""
    se_se_kernel(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires 3 hyperparameters.
Likely created using se_se_kernel_base() as an input.
Use with include("src/kernels/se_se_kernel.jl").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `["λ1", "λ2", "sratio"]`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
"""
function se_se_kernel(
    hyperparameters::AbstractVector{<:Real},
    δ::Real,
    dorder::AbstractVector{<:Integer})

    @assert length(hyperparameters)==3 "hyperparameters is the wrong length"
    dorder_len = 5
    @assert length(dorder)==dorder_len "dorder is the wrong length"
    @assert maximum(dorder) < 5 "No more than two derivatives for each time or hyperparameter can be calculated"
    @assert minimum(dorder) >= 0 "No integrals"

    dorder2 = dorder[2]
    dorder[2] += dorder[1]

    dorder_view = view(dorder, 2:dorder_len)
    
    λ1 = hyperparameters[1]
    λ2 = hyperparameters[2]
    sratio = hyperparameters[3]

    if dorder_view==[6, 0, 0, 2]
        func = -30*exp((-1/2)*δ^2/λ2^2)/λ2^6 + 90*exp((-1/2)*δ^2/λ2^2)*δ^2/λ2^8 - 30*exp((-1/2)*δ^2/λ2^2)*δ^4/λ2^10 + 2*exp((-1/2)*δ^2/λ2^2)*δ^6/λ2^12
    end

    if dorder_view==[5, 0, 0, 2]
        func = -30*exp((-1/2)*δ^2/λ2^2)*δ/λ2^6 + 20*exp((-1/2)*δ^2/λ2^2)*δ^3/λ2^8 - 2*exp((-1/2)*δ^2/λ2^2)*δ^5/λ2^10
    end

    if dorder_view==[4, 0, 0, 2]
        func = 6*exp((-1/2)*δ^2/λ2^2)/λ2^4 - 12*exp((-1/2)*δ^2/λ2^2)*δ^2/λ2^6 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4/λ2^8
    end

    if dorder_view==[3, 0, 0, 2]
        func = 6*exp((-1/2)*δ^2/λ2^2)*δ/λ2^4 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3/λ2^6
    end

    if dorder_view==[2, 0, 0, 2]
        func = -2*exp((-1/2)*δ^2/λ2^2)/λ2^2 + 2*exp((-1/2)*δ^2/λ2^2)*δ^2/λ2^4
    end

    if dorder_view==[1, 0, 0, 2]
        func = -2*exp((-1/2)*δ^2/λ2^2)*δ/λ2^2
    end

    if dorder_view==[0, 0, 0, 2]
        func = 2*exp((-1/2)*δ^2/λ2^2)
    end

    if dorder_view==[6, 0, 1, 1]
        func = 180*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^7 - 750*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^9 + 390*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^11 - 54*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio/λ2^13 + 2*exp((-1/2)*δ^2/λ2^2)*δ^8*sratio/λ2^15
    end

    if dorder_view==[5, 0, 1, 1]
        func = 180*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^7 - 190*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^9 + 40*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio/λ2^11 - 2*exp((-1/2)*δ^2/λ2^2)*δ^7*sratio/λ2^13
    end

    if dorder_view==[4, 0, 1, 1]
        func = -24*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^5 + 78*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^7 - 28*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^9 + 2*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio/λ2^11
    end

    if dorder_view==[3, 0, 1, 1]
        func = -24*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^5 + 18*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^7 - 2*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio/λ2^9
    end

    if dorder_view==[2, 0, 1, 1]
        func = 4*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^3 - 10*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^5 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^7
    end

    if dorder_view==[1, 0, 1, 1]
        func = 4*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^3 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^5
    end

    if dorder_view==[0, 0, 1, 1]
        func = 2*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^3
    end

    if dorder_view==[6, 1, 0, 1]
        func = 0
    end

    if dorder_view==[5, 1, 0, 1]
        func = 0
    end

    if dorder_view==[4, 1, 0, 1]
        func = 0
    end

    if dorder_view==[3, 1, 0, 1]
        func = 0
    end

    if dorder_view==[2, 1, 0, 1]
        func = 0
    end

    if dorder_view==[1, 1, 0, 1]
        func = 0
    end

    if dorder_view==[0, 1, 0, 1]
        func = 0
    end

    if dorder_view==[6, 0, 0, 1]
        func = -30*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^6 + 90*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^8 - 30*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^10 + 2*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio/λ2^12
    end

    if dorder_view==[5, 0, 0, 1]
        func = -30*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^6 + 20*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^8 - 2*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio/λ2^10
    end

    if dorder_view==[4, 0, 0, 1]
        func = 6*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^4 - 12*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^6 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^8
    end

    if dorder_view==[3, 0, 0, 1]
        func = 6*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^4 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^6
    end

    if dorder_view==[2, 0, 0, 1]
        func = -2*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^2 + 2*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^4
    end

    if dorder_view==[1, 0, 0, 1]
        func = -2*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^2
    end

    if dorder_view==[0, 0, 0, 1]
        func = 2*exp((-1/2)*δ^2/λ2^2)*sratio
    end

    if dorder_view==[6, 0, 2, 0]
        func = -630*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^8 + 3465*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^10 - 2520*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^12 + 546*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^14 - 42*exp((-1/2)*δ^2/λ2^2)*δ^8*sratio^2/λ2^16 + exp((-1/2)*δ^2/λ2^2)*δ^10*sratio^2/λ2^18
    end

    if dorder_view==[5, 0, 2, 0]
        func = -630*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^8 + 945*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^10 - 315*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^12 + 33*exp((-1/2)*δ^2/λ2^2)*δ^7*sratio^2/λ2^14 - exp((-1/2)*δ^2/λ2^2)*δ^9*sratio^2/λ2^16
    end

    if dorder_view==[4, 0, 2, 0]
        func = 60*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^6 - 285*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^8 + 165*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^10 - 25*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^12 + exp((-1/2)*δ^2/λ2^2)*δ^8*sratio^2/λ2^14
    end

    if dorder_view==[3, 0, 2, 0]
        func = 60*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^6 - 75*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^8 + 18*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^10 - exp((-1/2)*δ^2/λ2^2)*δ^7*sratio^2/λ2^12
    end

    if dorder_view==[2, 0, 2, 0]
        func = -6*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^4 + 27*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^6 - 12*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^8 + exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^10
    end

    if dorder_view==[1, 0, 2, 0]
        func = -6*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^4 + 7*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^6 - exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^8
    end

    if dorder_view==[0, 0, 2, 0]
        func = -3*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^4 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^6
    end

    if dorder_view==[6, 1, 1, 0]
        func = 0
    end

    if dorder_view==[5, 1, 1, 0]
        func = 0
    end

    if dorder_view==[4, 1, 1, 0]
        func = 0
    end

    if dorder_view==[3, 1, 1, 0]
        func = 0
    end

    if dorder_view==[2, 1, 1, 0]
        func = 0
    end

    if dorder_view==[1, 1, 1, 0]
        func = 0
    end

    if dorder_view==[0, 1, 1, 0]
        func = 0
    end

    if dorder_view==[6, 0, 1, 0]
        func = 90*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^7 - 375*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^9 + 195*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^11 - 27*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^13 + exp((-1/2)*δ^2/λ2^2)*δ^8*sratio^2/λ2^15
    end

    if dorder_view==[5, 0, 1, 0]
        func = 90*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^7 - 95*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^9 + 20*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^11 - exp((-1/2)*δ^2/λ2^2)*δ^7*sratio^2/λ2^13
    end

    if dorder_view==[4, 0, 1, 0]
        func = -12*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^5 + 39*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^7 - 14*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^9 + exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^11
    end

    if dorder_view==[3, 0, 1, 0]
        func = -12*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^5 + 9*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^7 - exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^9
    end

    if dorder_view==[2, 0, 1, 0]
        func = 2*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^3 - 5*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^5 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^7
    end

    if dorder_view==[1, 0, 1, 0]
        func = 2*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^3 - exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^5
    end

    if dorder_view==[0, 0, 1, 0]
        func = exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^3
    end

    if dorder_view==[6, 2, 0, 0]
        func = -630*exp((-1/2)*δ^2/λ1^2)/λ1^8 + 3465*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^10 - 2520*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^12 + 546*exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^14 - 42*exp((-1/2)*δ^2/λ1^2)*δ^8/λ1^16 + exp((-1/2)*δ^2/λ1^2)*δ^10/λ1^18
    end

    if dorder_view==[5, 2, 0, 0]
        func = -630*exp((-1/2)*δ^2/λ1^2)*δ/λ1^8 + 945*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^10 - 315*exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^12 + 33*exp((-1/2)*δ^2/λ1^2)*δ^7/λ1^14 - exp((-1/2)*δ^2/λ1^2)*δ^9/λ1^16
    end

    if dorder_view==[4, 2, 0, 0]
        func = 60*exp((-1/2)*δ^2/λ1^2)/λ1^6 - 285*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^8 + 165*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^10 - 25*exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^12 + exp((-1/2)*δ^2/λ1^2)*δ^8/λ1^14
    end

    if dorder_view==[3, 2, 0, 0]
        func = 60*exp((-1/2)*δ^2/λ1^2)*δ/λ1^6 - 75*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^8 + 18*exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^10 - exp((-1/2)*δ^2/λ1^2)*δ^7/λ1^12
    end

    if dorder_view==[2, 2, 0, 0]
        func = -6*exp((-1/2)*δ^2/λ1^2)/λ1^4 + 27*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^6 - 12*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^8 + exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^10
    end

    if dorder_view==[1, 2, 0, 0]
        func = -6*exp((-1/2)*δ^2/λ1^2)*δ/λ1^4 + 7*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^6 - exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^8
    end

    if dorder_view==[0, 2, 0, 0]
        func = -3*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^4 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^6
    end

    if dorder_view==[6, 1, 0, 0]
        func = 90*exp((-1/2)*δ^2/λ1^2)/λ1^7 - 375*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^9 + 195*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^11 - 27*exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^13 + exp((-1/2)*δ^2/λ1^2)*δ^8/λ1^15
    end

    if dorder_view==[5, 1, 0, 0]
        func = 90*exp((-1/2)*δ^2/λ1^2)*δ/λ1^7 - 95*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^9 + 20*exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^11 - exp((-1/2)*δ^2/λ1^2)*δ^7/λ1^13
    end

    if dorder_view==[4, 1, 0, 0]
        func = -12*exp((-1/2)*δ^2/λ1^2)/λ1^5 + 39*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^7 - 14*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^9 + exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^11
    end

    if dorder_view==[3, 1, 0, 0]
        func = -12*exp((-1/2)*δ^2/λ1^2)*δ/λ1^5 + 9*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^7 - exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^9
    end

    if dorder_view==[2, 1, 0, 0]
        func = 2*exp((-1/2)*δ^2/λ1^2)/λ1^3 - 5*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^5 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^7
    end

    if dorder_view==[1, 1, 0, 0]
        func = 2*exp((-1/2)*δ^2/λ1^2)*δ/λ1^3 - exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^5
    end

    if dorder_view==[0, 1, 0, 0]
        func = exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^3
    end

    if dorder_view==[6, 0, 0, 0]
        func = -15*exp((-1/2)*δ^2/λ1^2)/λ1^6 + 45*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^8 - 15*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^10 + exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^12 - 15*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^6 + 45*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^8 - 15*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^10 + exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^12
    end

    if dorder_view==[5, 0, 0, 0]
        func = -15*exp((-1/2)*δ^2/λ1^2)*δ/λ1^6 + 10*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^8 - exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^10 - 15*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^6 + 10*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^8 - exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^10
    end

    if dorder_view==[4, 0, 0, 0]
        func = 3*exp((-1/2)*δ^2/λ1^2)/λ1^4 - 6*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^6 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^8 + 3*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^4 - 6*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^6 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^8
    end

    if dorder_view==[3, 0, 0, 0]
        func = 3*exp((-1/2)*δ^2/λ1^2)*δ/λ1^4 - exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^6 + 3*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^4 - exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^6
    end

    if dorder_view==[2, 0, 0, 0]
        func = -exp((-1/2)*δ^2/λ1^2)/λ1^2 + exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^4 - exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^2 + exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^4
    end

    if dorder_view==[1, 0, 0, 0]
        func = -exp((-1/2)*δ^2/λ1^2)*δ/λ1^2 - exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^2
    end

    if dorder_view==[0, 0, 0, 0]
        func = exp((-1/2)*δ^2/λ2^2)*sratio^2 + exp((-1/2)*δ^2/λ1^2)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return se_se_kernel, 3  # the function handle and the number of kernel hyperparameters
