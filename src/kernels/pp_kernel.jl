import GPLinearODEMaker.powers_of_negative_one

"""
    pp_kernel(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires 1 hyperparameters.
Likely created using pp_kernel_base() as an input.
Use with include("src/kernels/pp_kernel.jl").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `["λ"]`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
"""
function pp_kernel(
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
    
    λ = hyperparameters[1]

    if abs(δ) > λ
        dorder[2] = dorder2
        return 0
    end

    dabs_δ = powers_of_negative_one(δ < 0)  # store derivative of abs()
    abs_δ = abs(δ)

    if dorder_view==[6, 2]
        func = -403200*abs_δ*dabs_δ^6/λ^9 + 1209600*dabs_δ^6*(1 - abs_δ/λ)/λ^8 - 21600*dabs_δ^5*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^7 + 7200*dabs_δ^5*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^6 - 720*dabs_δ^5*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^5
    end

    if dorder_view==[5, 2]
        func = -19200*abs_δ^2*dabs_δ^5/λ^9 - 288000*dabs_δ^5*(1 - abs_δ/λ)^2/λ^7 - 120*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ^5/λ^5 + 1200*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^5/λ^6 - 3600*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^5/λ^7 + 230400*abs_δ*dabs_δ^5*(1 - abs_δ/λ)/λ^8 - 6000*abs_δ*dabs_δ^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^7 + 1200*abs_δ*dabs_δ^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^6 + 12000*dabs_δ^4*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 4800*dabs_δ^4*(1 - abs_δ/λ)*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^5 + 600*dabs_δ^4*(1 - abs_δ/λ)*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^4
    end

    if dorder_view==[4, 2]
        func = 38400*dabs_δ^4*(1 - abs_δ/λ)^3/λ^6 + 11520*abs_δ^2*dabs_δ^4*(1 - abs_δ/λ)/λ^8 - 57600*abs_δ*dabs_δ^4*(1 - abs_δ/λ)^2/λ^7 - 480*abs_δ^2*dabs_δ^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^7 - 2880*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 1440*dabs_δ^3*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^3 + 120*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ^4*(1 - abs_δ/λ)/λ^4 + 240*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ^4/λ^6 - 960*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^4*(1 - abs_δ/λ)/λ^5 - 1200*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^4/λ^7 + 2400*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^4*(1 - abs_δ/λ)/λ^6 + 3840*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 960*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^5
    end

    if dorder_view==[3, 2]
        func = -2880*dabs_δ^3*(1 - abs_δ/λ)^4/λ^5 - 2880*abs_δ^2*dabs_δ^3*(1 - abs_δ/λ)^2/λ^7 + 7680*abs_δ*dabs_δ^3*(1 - abs_δ/λ)^3/λ^6 + 360*dabs_δ^2*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 240*dabs_δ^2*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3 + 60*dabs_δ^2*(1 - abs_δ/λ)^3*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^2 - 60*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3 + 360*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^4 - 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*dabs_δ^3/λ^7 - 720*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^5 + 360*abs_δ^2*dabs_δ^2*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 1080*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 360*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^4 - 240*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^5 + 960*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^6
    end

    if dorder_view==[2, 2]
        func = 96*dabs_δ^2*(1 - abs_δ/λ)^5/λ^4 + 320*abs_δ^2*dabs_δ^2*(1 - abs_δ/λ)^3/λ^6 - 480*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^4/λ^5 - 20*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 20*dabs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ + 20*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^2 - 80*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^3 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^4 - 120*abs_δ^2*dabs_δ*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 160*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 80*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3 + 120*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2/λ^4 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*dabs_δ^2*(1 - abs_δ/λ)/λ^6 - 360*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2/λ^5
    end

    if dorder_view==[1, 2]
        func = (1 - abs_δ/λ)^5*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4) + 20*abs_δ^2*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 10*abs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 10*abs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 5*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ*(1 - abs_δ/λ)^4/λ + 10*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ*(1 - abs_δ/λ)^4/λ^2 - 10*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ^3 - 40*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^3 - 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*dabs_δ*(1 - abs_δ/λ)^2/λ^5 + 80*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^4
    end

    if dorder_view==[0, 2]
        func = (48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*(1 - abs_δ/λ)^5 + 10*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*(1 - abs_δ/λ)^4/λ^2 + 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*(1 - abs_δ/λ)^3/λ^4 - 10*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^4/λ^3
    end

    if dorder_view==[6, 1]
        func = 28800*abs_δ*dabs_δ^6/λ^8 - 172800*dabs_δ^6*(1 - abs_δ/λ)/λ^7 + 3600*dabs_δ^5*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 720*dabs_δ^5*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^5
    end

    if dorder_view==[5, 1]
        func = 48000*dabs_δ^5*(1 - abs_δ/λ)^2/λ^6 - 120*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^5/λ^5 + 600*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^5/λ^6 - 19200*abs_δ*dabs_δ^5*(1 - abs_δ/λ)/λ^7 + 600*abs_δ*dabs_δ^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 2400*dabs_δ^4*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 600*dabs_δ^4*(1 - abs_δ/λ)*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^4
    end

    if dorder_view==[4, 1]
        func = -7680*dabs_δ^4*(1 - abs_δ/λ)^3/λ^5 + 5760*abs_δ*dabs_δ^4*(1 - abs_δ/λ)^2/λ^6 + 720*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3 + 120*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^4*(1 - abs_δ/λ)/λ^4 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^4/λ^6 - 480*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^4*(1 - abs_δ/λ)/λ^5 - 480*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5
    end

    if dorder_view==[3, 1]
        func = 720*dabs_δ^3*(1 - abs_δ/λ)^4/λ^4 - 960*abs_δ*dabs_δ^3*(1 - abs_δ/λ)^3/λ^5 - 120*dabs_δ^2*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 60*dabs_δ^2*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 60*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3 + 180*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^4 + 180*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^5
    end

    if dorder_view==[2, 1]
        func = -32*dabs_δ^2*(1 - abs_δ/λ)^5/λ^3 + 80*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^4/λ^4 + 10*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ + 20*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^2 - 40*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^3 - 40*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^2*(1 - abs_δ/λ)^2/λ^4
    end

    if dorder_view==[1, 1]
        func = (1 - abs_δ/λ)^5*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3) + 5*abs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 5*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ*(1 - abs_δ/λ)^4/λ + 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ^2 - 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^3
    end

    if dorder_view==[0, 1]
        func = (-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)^5 + 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^4/λ^2
    end

    if dorder_view==[6, 0]
        func = 28800*dabs_δ^6*(1 - abs_δ/λ)/λ^6 - 720*dabs_δ^5*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5
    end

    if dorder_view==[5, 0]
        func = -9600*dabs_δ^5*(1 - abs_δ/λ)^2/λ^5 - 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^5/λ^5 + 600*dabs_δ^4*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4
    end

    if dorder_view==[4, 0]
        func = 1920*dabs_δ^4*(1 - abs_δ/λ)^3/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^4*(1 - abs_δ/λ)/λ^4
    end

    if dorder_view==[3, 0]
        func = -240*dabs_δ^3*(1 - abs_δ/λ)^4/λ^3 + 60*dabs_δ^2*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3
    end

    if dorder_view==[2, 0]
        func = 16*dabs_δ^2*(1 - abs_δ/λ)^5/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ + 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^2*(1 - abs_δ/λ)^3/λ^2
    end

    if dorder_view==[1, 0]
        func = (1 - abs_δ/λ)^5*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2) - 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ
    end

    if dorder_view==[0, 0]
        func = (1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)^5
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return pp_kernel, 1  # the function handle and the number of kernel hyperparameters
