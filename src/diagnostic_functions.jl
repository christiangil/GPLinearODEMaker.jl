"""
    est_dΣdθ(glo, kernel_hyperparameters; return_est=true, return_anal=false, return_dif=false, return_bool=false, dif=1e-6, print_stuff=true)

Estimate the covariance derivatives of `glo` (a GLOM model) at
`kernel_hyperparameters`with forward differences

# Keyword Arguments
- `return_est::Bool=true`: Add the numerically estimated dΣdθs to the output
- `return_anal::Bool=false`: Add the analytical dΣdθs to the output
- `return_dif::Bool=false`: Add the difference between the dΣdθs to the output
- `return_bool::Bool=false`: Return just a similarity Boolean
- `dif::Real=1e-6`: The step size used in the forward difference method
- `print_stuff::Bool=true`: Whether to print things

# Output
- If `return_bool==true`, returns a Boolean for whether the analytical and
numerical estimates for the dΣdθs are approximately the same.
- Else returns a vector with some combination of the numerically estimated
dΣdθs, analytical dΣdθs, and differences between them
"""
function est_dΣdθ(glo::GLO, kernel_hyperparameters::Vector{T}; return_est::Bool=true, return_anal::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(glo.a0)), kernel_hyperparameters)

    if print_stuff
        println()
        println("est_dΣdθ: Check that our analytical dΣdθ are close to numerical estimates")
        println("hyperparameters: ", total_hyperparameters)
    end

    x = glo.x_obs
    return_vec = []
    coeff_orders = coefficient_orders(glo.n_out, glo.n_dif, a=glo.a0)

    # construct estimated dΣdθs
    if return_est || return_dif || return_bool
        val = covariance(glo, total_hyperparameters)
        est_dΣdθs = zeros(length(total_hyperparameters), glo.n_out * length(x), glo.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            if total_hyperparameters[i]!=0
                hold = copy(total_hyperparameters)
                hold[i] += dif
                est_dΣdθs[i, :, :] =  (covariance(glo, hold) - val) / dif
            end
        end
        if return_est; append!(return_vec, [est_dΣdθs]) end
    end

    # construct analytical dΣdθs
    if return_anal || return_dif || return_bool
        anal_dΣdθs = zeros(length(total_hyperparameters), glo.n_out * length(x), glo.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            anal_dΣdθs[i, :, :] =  covariance(glo, total_hyperparameters; dΣdθs_total=[i])
        end
        if return_anal; append!(return_vec, [anal_dΣdθs]) end
    end

    if return_dif || return_bool
        difs = est_dΣdθs - anal_dΣdθs
        append!(return_vec, [difs])
    end

    # find whether the analytical and estimated dΣdθs are approximately the same
    if return_bool
        no_differences = true
        min_thres = 5e-3
        max_val = 0
        for ind in 1:length(total_hyperparameters)
            est = est_dΣdθs[ind,:,:]
            est[est .== 0] .= 1e-8
            val = mean(abs.(difs[ind,:,:] ./ est))
            max_val = maximum([max_val, val])
            if val>min_thres
                no_differences = false
                if print_stuff; println("dΣ/dθ$ind has a average ratioed difference of: ", val) end
            end
        end
        if print_stuff; println("Maximum average dΣ/dθ ratioed difference of: ", max_val) end
        return no_differences
    end

    return return_vec

end

"""
    est_∇(f::Function, inputs::Vector{<:Real}; dif=1e-7, ignore_0_inputs=false)

Estimate the gradient of `f` at `inputs` with forward differences

# Examples
```jldoctest
julia> f(x) = x[1] + x[2]^2;

julia> isapprox([1,4], GPLinearODEMaker.est_∇(f, [2, 2.]); rtol=1e-5)
true
```
"""
function est_∇(f::Function, inputs::Vector{<:Real}; dif::Real=1e-7, ignore_0_inputs::Bool=false)
    # original value
    val = f(inputs)

    #estimate gradient
    j = 1
    if ignore_0_inputs
        grad = zeros(length(remove_zeros(inputs)))
    else
        grad = zeros(length(inputs))
    end
    for i in 1:length(inputs)
        if !ignore_0_inputs || inputs[i]!=0
            hold = copy(inputs)
            hold[i] += dif
            grad[j] =  (f(hold) - val) / dif
            j += 1
        end
    end
    return grad
end


"""
    est_∇nlogL_GLOM(glo, total_hyperparameters; dif=1e-7)

Estimate the gradient of `nlogL_GLOM(glo, total_hyperparameters)` with forward
differences
"""
function est_∇nlogL_GLOM(glo::GLO, total_hyperparameters::Vector{T}; dif::Real=1e-7) where {T<:Real}
    f(inputs) = nlogL_GLOM(glo, inputs)
    return est_∇(f, total_hyperparameters; dif=dif, ignore_0_inputs=true)
end

"""
    test_∇(est_G, G; print_stuff=true, function_name="function")

Check if two gradient vectors (`est_G` and `G`) are approximately the same

# Examples
```jldoctest
julia> GPLinearODEMaker.test_∇([1., 2, 3], [1.0001, 2, 3]; print_stuff=false)
true
julia> GPLinearODEMaker.test_∇([1., 2, 3], [0., 2, 3]; print_stuff=false)
false
```
"""
function test_∇(est_G::Vector{T}, G::Vector{T}; print_stuff::Bool=true, function_name::String="function") where {T<:Real}

    if print_stuff
        println()
        println("test_∇: Check that our analytical $function_name is close to numerical estimates")
        # println("only values for non-zero hyperparameters are shown!")
        # println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(G)
        if !isapprox(G[i], est_G[i], rtol=5e-2)
            if print_stuff
                println("mismatch df/dθ" * string(i))
            end
            no_mismatch = false
        end
    end

    return no_mismatch
end

"""
    test_∇nlogL_GLOM(glo, kernel_hyperparameters; dif=1e-4, print_stuff=true)

Check if `∇nlogL_GLOM(glo, total_hyperparameters)` is close to numerical
estimates provided by `est_∇nlogL_GLOM(glo, total_hyperparameters)`
"""
function test_∇nlogL_GLOM(glo::GLO, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}
    total_hyperparameters = append!(collect(Iterators.flatten(glo.a0)), kernel_hyperparameters)
    return test_∇(est_∇nlogL_GLOM(glo, total_hyperparameters; dif=dif),
        ∇nlogL_GLOM(glo, total_hyperparameters);
        print_stuff=print_stuff, function_name="∇nlogL_GLOM")
end

"""
    est_∇∇(g::Function, inputs::Vector{<:Real}; dif=1e-7, ignore_0_inputs=false)

Estimate the Hessian of a function whose gradients are provided by `g` at
`inputs` with forward differences

# Examples
```jldoctest
julia> g(x) = [(x[2] ^ 2) / 2, x[1] * x[2]];  # h(x) = [0 x[2], x[2] x[1]]

julia> isapprox([0. 9; 9 4], GPLinearODEMaker.est_∇∇(g, [4., 9]); rtol=1e-5)
true
```
"""
function est_∇∇(g::Function, inputs::Vector{<:Real}; dif::Real=1e-7, ignore_0_inputs::Bool=false)

    val = g(inputs)

    #estimate hessian
    j = 1
    dim = length(remove_zeros(inputs))
    hess = zeros(dim, dim)
    for i in 1:length(inputs)
        if !ignore_0_inputs || inputs[i]!=0
            hold = copy(inputs)
            hold[i] += dif
            hess[j, :] =  (g(hold) - val) / dif
            j += 1
        end
    end
    return symmetric_A(hess)

end

"""
    est_∇∇_from_f(f::Function, inputs::Vector{<:Real}; dif=1e-7, ignore_0_inputs=false)

Estimate the Hessian of `f` at `inputs` with forward differences. WARNING: The
result is very sensitive to `dif`

# Examples
```jldoctest
julia> f(x) = (x[1] * x[2] ^ 2) / 2;  # h(x) = [0 x[2], x[2] x[1]]

julia> isapprox([0. 9; 9 4], GPLinearODEMaker.est_∇∇_from_f(f, [4., 9]; dif=1e-4); rtol=1e-3)
true
```
"""
function est_∇∇_from_f(f::Function, inputs::Vector{<:Real}; dif::Real=1e-7, ignore_0_inputs::Bool=false)

    val = est_∇(f, inputs; dif=dif, ignore_0_inputs=ignore_0_inputs)

    #estimate hessian
    j = 1
    hess = zeros(length(inputs), length(inputs))
    for i in 1:length(inputs)
        if !ignore_0_inputs || inputs[i]!=0
            hold = copy(inputs)
            hold[i] += dif
            hess[j, :] =  (est_∇(f, hold; dif=dif, ignore_0_inputs=ignore_0_inputs) - val) / dif
            j += 1
        end
    end
    return symmetric_A(hess)

end


"""
    est_∇∇nlogL_GLOM(glo, total_hyperparameters; dif=1e-4)

Estimate the Hessian of `nlogL_GLOM(glo, total_hyperparameters)` with forward
differences
"""
function est_∇∇nlogL_GLOM(glo::GLO, total_hyperparameters::Vector{T}; dif::Real=1e-4) where {T<:Real}

    g(inputs) = ∇nlogL_GLOM(glo, inputs)
    return est_∇∇(g, total_hyperparameters; ignore_0_inputs=true)

end


"""
    test_∇∇(est_H, H; print_stuff=true, function_name="function", rtol=1e-3)

Check if two Hessian matrices (`est_H` and `H`) are approximately the same

# Examples
```jldoctest
julia> GPLinearODEMaker.test_∇∇([1. 2; 3 4], [1.0001 2; 3 4]; print_stuff=false)
true
julia> GPLinearODEMaker.test_∇∇([1. 2; 3 4], [0. 2; 3 4]; print_stuff=false)
false
```
"""
function test_∇∇(est_H::Union{Symmetric{T,Matrix{T}},Matrix{T}}, H::Union{Symmetric{T,Matrix{T}},Matrix{T}}; print_stuff::Bool=true, function_name::String="function", rtol::Real=1e-3) where {T<:Real}

    if print_stuff
        println()
        println("test_∇∇: Check that our analytical $function_name is close to numerical estimates")
        # println("only values for non-zero hyperparameters are shown!")
        # println("hyperparameters: ", total_hyperparameters)
        println("analytical:")
        for i in 1:size(H, 1)
            println(H[i, :])
        end
        println("numerical:")
        for i in 1:size(est_H, 1)
            println(est_H[i, :])
        end
    end

    no_mismatch = true
    matches = fill(1, size(H))
    for i in 1:size(H, 1)
        for j in 1:size(H, 2)
            if !(isapprox(H[i, j], est_H[i, j], rtol=rtol))
                # println("mismatch d2nlogL/dθ" * string(i) * "dθ" * string(j))
                matches[i, j] = 0
                no_mismatch = false
            end
        end
    end

    if !no_mismatch && print_stuff
        println("mismatches at 0s")
        for i in 1:size(H, 1)
            println(matches[i, :])
        end
    end

    return no_mismatch
end


"""
    test_∇∇nlogL_GLOM(glo, kernel_hyperparameters; dif=1e-4, print_stuff=true)

Check if `∇∇nlogL_GLOM(glo, total_hyperparameters)` is close to numerical
estimates provided by `est_∇∇nlogL_GLOM(glo, total_hyperparameters)`
"""
function test_∇∇nlogL_GLOM(glo::GLO, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(glo.a0)), kernel_hyperparameters)
    H = ∇∇nlogL_GLOM(glo, total_hyperparameters)
    est_H = est_∇∇nlogL_GLOM(glo, total_hyperparameters; dif=dif)

    return test_∇∇(est_H, H; print_stuff=print_stuff, function_name="∇∇nlogL_GLOM")

end
