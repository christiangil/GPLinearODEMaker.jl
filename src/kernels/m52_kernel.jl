import GPLinearODEMaker.powers_of_negative_one

"""
m52_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using m52_kernel_base() as an input.
Use with include("src/kernels/m52_kernel.jl").
hyperparameters == ["λ"]
"""
function m52_kernel(
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

    dabs_δ = powers_of_negative_one(δ < 0)  # store derivative of abs()
    abs_δ = abs(δ)

    if dorder_view==[6, 2]
        func = 52500.0*exp(-2.23606797749979*abs_δ/λ)/λ^8 + 6250.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2/λ^10 - 39131.1896062463*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^9 + 5250.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^8 - 1500.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^7 - 10062.3058987491*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^7 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ^6 + 3354.10196624969*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^6 - 335.410196624969*exp(-2.23606797749979*abs_δ/λ)*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3)/λ^5 + 625.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^10 - 3913.11896062463*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^9 - 1677.05098312484*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^9 + 559.016994374948*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^8 + 9000.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^8 - 1500.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^7
    end

    if dorder_view==[5, 2]
        func = -11180.339887499*exp(-2.23606797749979*abs_δ/λ)/λ^7 - 1863.38998124983*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2/λ^9 + 10000.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^8 - 1677.05098312484*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^7 + 559.016994374948*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^6 + 2500.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^6 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ)*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ^5 - 1000.0*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^5 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3)/λ^4 - 279.508497187474*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^9 + 1500.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^8 + 625.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^8 - 250.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^7 - 2795.08497187474*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^7 + 559.016994374948*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^6
    end

    if dorder_view==[4, 2]
        func = 2000.0*exp(-2.23606797749979*abs_δ/λ)/λ^6 + 500.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2/λ^8 - 2236.06797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^7 + 500.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 - 200.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^5 + 25.0*exp(-2.23606797749979*abs_δ/λ)*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ^4 + 125.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^8 - 559.016994374948*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^7 + 111.80339887499*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^6 - 536.65631459995*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5 + 268.328157299975*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3)/λ^3 - 223.606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^7 + 800.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^6 - 200.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^5
    end

    if dorder_view==[3, 2]
        func = -268.328157299975*exp(-2.23606797749979*abs_δ/λ)*dabs_δ/λ^5 + 90.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4 - 60.0*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^3 + 15.0*exp(-2.23606797749979*abs_δ/λ)*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3)/λ^2 - 111.80339887499*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*dabs_δ/λ^7 + 400.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ/λ^6 + 75.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^6 - 201.246117974981*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5 - 134.164078649987*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^4 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ^3 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^7 + 200.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 - 50.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^5
    end

    if dorder_view==[2, 2]
        func = 20.0*exp(-2.23606797749979*abs_δ/λ)/λ^4 + 16.6666666666667*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2/λ^6 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^5 + 30.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 - 20.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^3 + 5.0*exp(-2.23606797749979*abs_δ/λ)*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ^2 + 25.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 - 67.0820393249937*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5 + 22.3606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^4 - 8.94427190999916*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^3 + 8.94427190999916*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3)/λ - 22.3606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5 + 40.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4 - 20.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^3
    end

    if dorder_view==[1, 2]
        func = exp(-2.23606797749979*abs_δ/λ)*(16.6666666666667*abs_δ*dabs_δ/λ^4 + 4.47213595499958*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^3) + 5.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^3 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3)/λ - 11.180339887499*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5 + 20.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 - 10.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^3
    end

    if dorder_view==[0, 2]
        func = exp(-2.23606797749979*abs_δ/λ)*(6.66666666666667*abs_δ^2/λ^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^3) + 5.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^2
    end

    if dorder_view==[6, 1]
        func = -7500.0*exp(-2.23606797749979*abs_δ/λ)/λ^7 + 2795.08497187474*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^8 - 750.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^7 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^6 + 1677.05098312484*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^6 - 335.410196624969*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^5 + 279.508497187474*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^8 - 750.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^7
    end

    if dorder_view==[5, 1]
        func = 1863.38998124983*exp(-2.23606797749979*abs_δ/λ)/λ^6 - 833.333333333334*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^7 + 279.508497187474*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^5 - 500.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^4 - 125.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^7 + 279.508497187474*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^6
    end

    if dorder_view==[4, 1]
        func = -400.0*exp(-2.23606797749979*abs_δ/λ)/λ^5 + 223.606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^6 - 100.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5 + 25.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^4 + 55.9016994374948*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^3 - 100.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5
    end

    if dorder_view==[3, 1]
        func = 67.0820393249937*exp(-2.23606797749979*abs_δ/λ)*dabs_δ/λ^4 - 30.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^3 + 15.0*exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ^2 - 50.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ/λ^5 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^3 - 25.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5
    end

    if dorder_view==[2, 1]
        func = -6.66666666666667*exp(-2.23606797749979*abs_δ/λ)/λ^3 + 7.4535599249993*exp(-2.23606797749979*abs_δ/λ)*abs_δ/λ^4 - 10.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^3 + 5.0*exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ^2 + 11.180339887499*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2)/λ - 10.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^3
    end

    if dorder_view==[1, 1]
        func = exp(-2.23606797749979*abs_δ/λ)*(-5.0*abs_δ*dabs_δ/λ^3 - 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^2 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2)/λ - 5.0*exp(-2.23606797749979*abs_δ/λ)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^3
    end

    if dorder_view==[0, 1]
        func = exp(-2.23606797749979*abs_δ/λ)*(-1.66666666666667*abs_δ^2/λ^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^2
    end

    if dorder_view==[6, 0]
        func = 1250.0*exp(-2.23606797749979*abs_δ/λ)/λ^6 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^6 - 335.410196624969*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^5
    end

    if dorder_view==[5, 0]
        func = -372.677996249965*exp(-2.23606797749979*abs_δ/λ)/λ^5 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^5 + 125.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^4
    end

    if dorder_view==[4, 0]
        func = 100.0*exp(-2.23606797749979*abs_δ/λ)/λ^4 + 25.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^3
    end

    if dorder_view==[3, 0]
        func = -22.3606797749979*exp(-2.23606797749979*abs_δ/λ)*dabs_δ/λ^3 + 15.0*exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ^2 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^3
    end

    if dorder_view==[2, 0]
        func = 3.33333333333333*exp(-2.23606797749979*abs_δ/λ)/λ^2 + 5.0*exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ)/λ
    end

    if dorder_view==[1, 0]
        func = exp(-2.23606797749979*abs_δ/λ)*(1.66666666666667*abs_δ*dabs_δ/λ^2 + 2.23606797749979*(1 + 0.74535599249993*abs_δ/λ)*dabs_δ/λ) - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)/λ
    end

    if dorder_view==[0, 0]
        func = exp(-2.23606797749979*abs_δ/λ)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ)/λ)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return m52_kernel, 1  # the function handle and the number of kernel hyperparameters
