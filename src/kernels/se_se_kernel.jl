import GPLinearODEMaker.powers_of_negative_one

"""
se_se_kernel function created by kernel_coder(). Requires 3 hyperparameters. Likely created using se_se_kernel_base() as an input.
Use with include("src/kernels/se_se_kernel.jl").
hyperparameters == ["λ1", "λ2", "sratio"]
"""
function se_se_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==3 "hyperparameters is the wrong length"
    dorder_len = 5
    @assert length(dorder)==dorder_len "dorder is the wrong length"
    dorder2 = dorder[2]
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder[2] = sum(dorder[1:2])

    λ1 = hyperparameters[1]
    λ2 = hyperparameters[2]
    sratio = hyperparameters[3]

    if view(dorder, 2:dorder_len)==[4, 0, 0, 2]
        func = 6*exp((-1/2)*δ^2/λ2^2)/λ2^4 - 12*exp((-1/2)*δ^2/λ2^2)*δ^2/λ2^6 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4/λ2^8
    end

    if view(dorder, 2:dorder_len)==[3, 0, 0, 2]
        func = 6*exp((-1/2)*δ^2/λ2^2)*δ/λ2^4 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3/λ2^6
    end

    if view(dorder, 2:dorder_len)==[2, 0, 0, 2]
        func = -2*exp((-1/2)*δ^2/λ2^2)/λ2^2 + 2*exp((-1/2)*δ^2/λ2^2)*δ^2/λ2^4
    end

    if view(dorder, 2:dorder_len)==[1, 0, 0, 2]
        func = -2*exp((-1/2)*δ^2/λ2^2)*δ/λ2^2
    end

    if view(dorder, 2:dorder_len)==[0, 0, 0, 2]
        func = 2*exp((-1/2)*δ^2/λ2^2)
    end

    if view(dorder, 2:dorder_len)==[4, 0, 1, 1]
        func = -24*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^5 + 78*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^7 - 28*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^9 + 2*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio/λ2^11
    end

    if view(dorder, 2:dorder_len)==[3, 0, 1, 1]
        func = -24*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^5 + 18*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^7 - 2*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio/λ2^9
    end

    if view(dorder, 2:dorder_len)==[2, 0, 1, 1]
        func = 4*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^3 - 10*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^5 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^7
    end

    if view(dorder, 2:dorder_len)==[1, 0, 1, 1]
        func = 4*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^3 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^5
    end

    if view(dorder, 2:dorder_len)==[0, 0, 1, 1]
        func = 2*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^3
    end

    if view(dorder, 2:dorder_len)==[4, 1, 0, 1]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[3, 1, 0, 1]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[2, 1, 0, 1]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[1, 1, 0, 1]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[0, 1, 0, 1]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[4, 0, 0, 1]
        func = 6*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^4 - 12*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^6 + 2*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio/λ2^8
    end

    if view(dorder, 2:dorder_len)==[3, 0, 0, 1]
        func = 6*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^4 - 2*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio/λ2^6
    end

    if view(dorder, 2:dorder_len)==[2, 0, 0, 1]
        func = -2*exp((-1/2)*δ^2/λ2^2)*sratio/λ2^2 + 2*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio/λ2^4
    end

    if view(dorder, 2:dorder_len)==[1, 0, 0, 1]
        func = -2*exp((-1/2)*δ^2/λ2^2)*δ*sratio/λ2^2
    end

    if view(dorder, 2:dorder_len)==[0, 0, 0, 1]
        func = 2*exp((-1/2)*δ^2/λ2^2)*sratio
    end

    if view(dorder, 2:dorder_len)==[4, 0, 2, 0]
        func = 60*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^6 - 285*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^8 + 165*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^10 - 25*exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^12 + exp((-1/2)*δ^2/λ2^2)*δ^8*sratio^2/λ2^14
    end

    if view(dorder, 2:dorder_len)==[3, 0, 2, 0]
        func = 60*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^6 - 75*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^8 + 18*exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^10 - exp((-1/2)*δ^2/λ2^2)*δ^7*sratio^2/λ2^12
    end

    if view(dorder, 2:dorder_len)==[2, 0, 2, 0]
        func = -6*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^4 + 27*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^6 - 12*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^8 + exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^10
    end

    if view(dorder, 2:dorder_len)==[1, 0, 2, 0]
        func = -6*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^4 + 7*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^6 - exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^8
    end

    if view(dorder, 2:dorder_len)==[0, 0, 2, 0]
        func = -3*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^4 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^6
    end

    if view(dorder, 2:dorder_len)==[4, 1, 1, 0]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[3, 1, 1, 0]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[2, 1, 1, 0]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[1, 1, 1, 0]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[0, 1, 1, 0]
        func = 0
    end

    if view(dorder, 2:dorder_len)==[4, 0, 1, 0]
        func = -12*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^5 + 39*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^7 - 14*exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^9 + exp((-1/2)*δ^2/λ2^2)*δ^6*sratio^2/λ2^11
    end

    if view(dorder, 2:dorder_len)==[3, 0, 1, 0]
        func = -12*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^5 + 9*exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^7 - exp((-1/2)*δ^2/λ2^2)*δ^5*sratio^2/λ2^9
    end

    if view(dorder, 2:dorder_len)==[2, 0, 1, 0]
        func = 2*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^3 - 5*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^5 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^7
    end

    if view(dorder, 2:dorder_len)==[1, 0, 1, 0]
        func = 2*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^3 - exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^5
    end

    if view(dorder, 2:dorder_len)==[0, 0, 1, 0]
        func = exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^3
    end

    if view(dorder, 2:dorder_len)==[4, 2, 0, 0]
        func = 60*exp((-1/2)*δ^2/λ1^2)/λ1^6 - 285*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^8 + 165*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^10 - 25*exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^12 + exp((-1/2)*δ^2/λ1^2)*δ^8/λ1^14
    end

    if view(dorder, 2:dorder_len)==[3, 2, 0, 0]
        func = 60*exp((-1/2)*δ^2/λ1^2)*δ/λ1^6 - 75*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^8 + 18*exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^10 - exp((-1/2)*δ^2/λ1^2)*δ^7/λ1^12
    end

    if view(dorder, 2:dorder_len)==[2, 2, 0, 0]
        func = -6*exp((-1/2)*δ^2/λ1^2)/λ1^4 + 27*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^6 - 12*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^8 + exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^10
    end

    if view(dorder, 2:dorder_len)==[1, 2, 0, 0]
        func = -6*exp((-1/2)*δ^2/λ1^2)*δ/λ1^4 + 7*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^6 - exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^8
    end

    if view(dorder, 2:dorder_len)==[0, 2, 0, 0]
        func = -3*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^4 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^6
    end

    if view(dorder, 2:dorder_len)==[4, 1, 0, 0]
        func = -12*exp((-1/2)*δ^2/λ1^2)/λ1^5 + 39*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^7 - 14*exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^9 + exp((-1/2)*δ^2/λ1^2)*δ^6/λ1^11
    end

    if view(dorder, 2:dorder_len)==[3, 1, 0, 0]
        func = -12*exp((-1/2)*δ^2/λ1^2)*δ/λ1^5 + 9*exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^7 - exp((-1/2)*δ^2/λ1^2)*δ^5/λ1^9
    end

    if view(dorder, 2:dorder_len)==[2, 1, 0, 0]
        func = 2*exp((-1/2)*δ^2/λ1^2)/λ1^3 - 5*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^5 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^7
    end

    if view(dorder, 2:dorder_len)==[1, 1, 0, 0]
        func = 2*exp((-1/2)*δ^2/λ1^2)*δ/λ1^3 - exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^5
    end

    if view(dorder, 2:dorder_len)==[0, 1, 0, 0]
        func = exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^3
    end

    if view(dorder, 2:dorder_len)==[4, 0, 0, 0]
        func = 3*exp((-1/2)*δ^2/λ1^2)/λ1^4 - 6*exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^6 + exp((-1/2)*δ^2/λ1^2)*δ^4/λ1^8 + 3*exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^4 - 6*exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^6 + exp((-1/2)*δ^2/λ2^2)*δ^4*sratio^2/λ2^8
    end

    if view(dorder, 2:dorder_len)==[3, 0, 0, 0]
        func = 3*exp((-1/2)*δ^2/λ1^2)*δ/λ1^4 - exp((-1/2)*δ^2/λ1^2)*δ^3/λ1^6 + 3*exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^4 - exp((-1/2)*δ^2/λ2^2)*δ^3*sratio^2/λ2^6
    end

    if view(dorder, 2:dorder_len)==[2, 0, 0, 0]
        func = -exp((-1/2)*δ^2/λ1^2)/λ1^2 + exp((-1/2)*δ^2/λ1^2)*δ^2/λ1^4 - exp((-1/2)*δ^2/λ2^2)*sratio^2/λ2^2 + exp((-1/2)*δ^2/λ2^2)*δ^2*sratio^2/λ2^4
    end

    if view(dorder, 2:dorder_len)==[1, 0, 0, 0]
        func = -exp((-1/2)*δ^2/λ1^2)*δ/λ1^2 - exp((-1/2)*δ^2/λ2^2)*δ*sratio^2/λ2^2
    end

    if view(dorder, 2:dorder_len)==[0, 0, 0, 0]
        func = exp((-1/2)*δ^2/λ2^2)*sratio^2 + exp((-1/2)*δ^2/λ1^2)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return  powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return se_se_kernel, 3  # the function handle and the number of kernel hyperparameters
