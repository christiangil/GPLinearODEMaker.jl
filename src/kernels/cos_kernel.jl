import GPLinearODEMaker.powers_of_negative_one

"""
cos_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using cos_kernel_base() as an input.
Use with include("src/kernels/cos_kernel.jl").
hyperparameters == ["λ"]
"""
function cos_kernel(
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
        func = -2584214.15233042*cos(6.28318530717959*δ/λ)/λ^8 + 5412365.46417601*δ*sin(6.28318530717959*δ/λ)/λ^9 + 2429063.94011407*δ^2*cos(6.28318530717959*δ/λ)/λ^10
    end

    if dorder_view==[5, 2]
        func = -293778.89739387*sin(6.28318530717959*δ/λ)/λ^7 - 738346.900665833*δ*cos(6.28318530717959*δ/λ)/λ^8 + 386597.533155429*δ^2*sin(6.28318530717959*δ/λ)/λ^9
    end

    if dorder_view==[4, 2]
        func = 31170.9091308808*cos(6.28318530717959*δ/λ)/λ^6 - 97926.29913129*δ*sin(6.28318530717959*δ/λ)/λ^7 - 61528.9083888195*δ^2*cos(6.28318530717959*δ/λ)/λ^8
    end

    if dorder_view==[3, 2]
        func = 2976.60256130878*sin(6.28318530717959*δ/λ)/λ^5 + 12468.3636523523*δ*cos(6.28318530717959*δ/λ)/λ^6 - 9792.629913129*δ^2*sin(6.28318530717959*δ/λ)/λ^7
    end

    if dorder_view==[2, 2]
        func = -236.870505626145*cos(6.28318530717959*δ/λ)/λ^4 + 1488.30128065439*δ*sin(6.28318530717959*δ/λ)/λ^5 + 1558.54545654404*δ^2*cos(6.28318530717959*δ/λ)/λ^6
    end

    if dorder_view==[1, 2]
        func = -12.5663706143592*sin(6.28318530717959*δ/λ)/λ^3 - 157.91367041743*δ*cos(6.28318530717959*δ/λ)/λ^4 + 248.050213442399*δ^2*sin(6.28318530717959*δ/λ)/λ^5
    end

    if dorder_view==[0, 2]
        func = -12.5663706143592*δ*sin(6.28318530717959*δ/λ)/λ^3 - 39.4784176043574*δ^2*cos(6.28318530717959*δ/λ)/λ^4
    end

    if dorder_view==[6, 1]
        func = 369173.450332917*cos(6.28318530717959*δ/λ)/λ^7 - 386597.533155429*δ*sin(6.28318530717959*δ/λ)/λ^8
    end

    if dorder_view==[5, 1]
        func = 48963.149565645*sin(6.28318530717959*δ/λ)/λ^6 + 61528.9083888195*δ*cos(6.28318530717959*δ/λ)/λ^7
    end

    if dorder_view==[4, 1]
        func = -6234.18182617615*cos(6.28318530717959*δ/λ)/λ^5 + 9792.629913129*δ*sin(6.28318530717959*δ/λ)/λ^6
    end

    if dorder_view==[3, 1]
        func = -744.150640327196*sin(6.28318530717959*δ/λ)/λ^4 - 1558.54545654404*δ*cos(6.28318530717959*δ/λ)/λ^5
    end

    if dorder_view==[2, 1]
        func = 78.9568352087149*cos(6.28318530717959*δ/λ)/λ^3 - 248.050213442399*δ*sin(6.28318530717959*δ/λ)/λ^4
    end

    if dorder_view==[1, 1]
        func = 6.28318530717959*sin(6.28318530717959*δ/λ)/λ^2 + 39.4784176043574*δ*cos(6.28318530717959*δ/λ)/λ^3
    end

    if dorder_view==[0, 1]
        func = 6.28318530717959*δ*sin(6.28318530717959*δ/λ)/λ^2
    end

    if dorder_view==[6, 0]
        func = -61528.9083888195*cos(6.28318530717959*δ/λ)/λ^6
    end

    if dorder_view==[5, 0]
        func = -9792.629913129*sin(6.28318530717959*δ/λ)/λ^5
    end

    if dorder_view==[4, 0]
        func = 1558.54545654404*cos(6.28318530717959*δ/λ)/λ^4
    end

    if dorder_view==[3, 0]
        func = 248.050213442399*sin(6.28318530717959*δ/λ)/λ^3
    end

    if dorder_view==[2, 0]
        func = -39.4784176043574*cos(6.28318530717959*δ/λ)/λ^2
    end

    if dorder_view==[1, 0]
        func = -6.28318530717959*sin(6.28318530717959*δ/λ)/λ
    end

    if dorder_view==[0, 0]
        func = cos(6.28318530717959*δ/λ)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return cos_kernel, 1  # the function handle and the number of kernel hyperparameters
