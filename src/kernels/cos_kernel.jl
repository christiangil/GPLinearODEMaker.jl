import GPLinearODEMaker.powers_of_negative_one

"""
    cos_kernel(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires 1 hyperparameters.
Likely created using cos_kernel_base() as an input.
Use with include("src/kernels/cos_kernel.jl").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `["λ"]`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
"""
function cos_kernel(
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

    if dorder_view==[6, 2]
        func = -2688*cos(2*δ*π/λ)*π^6/λ^8 + 1792*δ*sin(2*δ*π/λ)*π^7/λ^9 + 256*δ^2*cos(2*δ*π/λ)*π^8/λ^10
    end

    if dorder_view==[5, 2]
        func = -960*sin(2*δ*π/λ)*π^5/λ^7 - 768*δ*cos(2*δ*π/λ)*π^6/λ^8 + 128*δ^2*sin(2*δ*π/λ)*π^7/λ^9
    end

    if dorder_view==[4, 2]
        func = 320*cos(2*δ*π/λ)*π^4/λ^6 - 320*δ*sin(2*δ*π/λ)*π^5/λ^7 - 64*δ^2*cos(2*δ*π/λ)*π^6/λ^8
    end

    if dorder_view==[3, 2]
        func = 96*sin(2*δ*π/λ)*π^3/λ^5 + 128*δ*cos(2*δ*π/λ)*π^4/λ^6 - 32*δ^2*sin(2*δ*π/λ)*π^5/λ^7
    end

    if dorder_view==[2, 2]
        func = -24*cos(2*δ*π/λ)*π^2/λ^4 + 48*δ*sin(2*δ*π/λ)*π^3/λ^5 + 16*δ^2*cos(2*δ*π/λ)*π^4/λ^6
    end

    if dorder_view==[1, 2]
        func = -4*sin(2*δ*π/λ)*π/λ^3 - 16*δ*cos(2*δ*π/λ)*π^2/λ^4 + 8*δ^2*sin(2*δ*π/λ)*π^3/λ^5
    end

    if dorder_view==[0, 2]
        func = -4*δ*sin(2*δ*π/λ)*π/λ^3 - 4*δ^2*cos(2*δ*π/λ)*π^2/λ^4
    end

    if dorder_view==[6, 1]
        func = 384*cos(2*δ*π/λ)*π^6/λ^7 - 128*δ*sin(2*δ*π/λ)*π^7/λ^8
    end

    if dorder_view==[5, 1]
        func = 160*sin(2*δ*π/λ)*π^5/λ^6 + 64*δ*cos(2*δ*π/λ)*π^6/λ^7
    end

    if dorder_view==[4, 1]
        func = -64*cos(2*δ*π/λ)*π^4/λ^5 + 32*δ*sin(2*δ*π/λ)*π^5/λ^6
    end

    if dorder_view==[3, 1]
        func = -24*sin(2*δ*π/λ)*π^3/λ^4 - 16*δ*cos(2*δ*π/λ)*π^4/λ^5
    end

    if dorder_view==[2, 1]
        func = 8*cos(2*δ*π/λ)*π^2/λ^3 - 8*δ*sin(2*δ*π/λ)*π^3/λ^4
    end

    if dorder_view==[1, 1]
        func = 2*sin(2*δ*π/λ)*π/λ^2 + 4*δ*cos(2*δ*π/λ)*π^2/λ^3
    end

    if dorder_view==[0, 1]
        func = 2*δ*sin(2*δ*π/λ)*π/λ^2
    end

    if dorder_view==[6, 0]
        func = -64*cos(2*δ*π/λ)*π^6/λ^6
    end

    if dorder_view==[5, 0]
        func = -32*sin(2*δ*π/λ)*π^5/λ^5
    end

    if dorder_view==[4, 0]
        func = 16*cos(2*δ*π/λ)*π^4/λ^4
    end

    if dorder_view==[3, 0]
        func = 8*sin(2*δ*π/λ)*π^3/λ^3
    end

    if dorder_view==[2, 0]
        func = -4*cos(2*δ*π/λ)*π^2/λ^2
    end

    if dorder_view==[1, 0]
        func = -2*sin(2*δ*π/λ)*π/λ
    end

    if dorder_view==[0, 0]
        func = cos(2*δ*π/λ)
    end

    dorder[2] = dorder2  # resetting dorder[2]
    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives

end


return cos_kernel, 1  # the function handle and the number of kernel hyperparameters
