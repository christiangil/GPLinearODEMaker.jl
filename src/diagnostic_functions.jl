"""
Estimate the covariance derivatives with forward differences
"""
function est_dΣdθ(prob_def::GLO, kernel_hyperparameters::Vector{T}; return_est::Bool=true, return_anal::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)

    if print_stuff
        println()
        println("est_dΣdθ: Check that our analytical dΣdθ are close to numerical estimates")
        println("hyperparameters: ", total_hyperparameters)
    end

    x = prob_def.x_obs
    return_vec = []
    coeff_orders = coefficient_orders(prob_def.n_out, prob_def.n_dif, a=prob_def.a0)

    # construct estimated dΣdθs
    if return_est || return_dif || return_bool
        val = covariance(prob_def, total_hyperparameters)
        est_dΣdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            if total_hyperparameters[i]!=0
                hold = copy(total_hyperparameters)
                hold[i] += dif
                est_dΣdθs[i, :, :] =  (covariance(prob_def, hold) - val) / dif
            end
        end
        if return_est; append!(return_vec, [est_dΣdθs]) end
    end

    # construct analytical dΣdθs
    if return_anal || return_dif || return_bool
        anal_dΣdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            anal_dΣdθs[i, :, :] =  covariance(prob_def, total_hyperparameters; dΣdθs_total=[i])
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

function est_∇(f::Function, inputs::Vector{<:Real}; dif::Real=1e-7, ignore_0_inputs::Bool=false)
    # original value
    val = f(inputs)

    #estimate gradient
    j = 1
    grad = zeros(length(remove_zeros(inputs)))
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
Estimate the gradient of nlogL_GLOM() with forward differences

Parameters:

prob_def (GLO): A structure that holds all of the relevant
    information for constructing the model used in the GLOM et al. 2017+ paper
total_hyperparameters (vector): The hyperparameters for the GP model, including
    both the coefficient hyperparameters and the kernel hyperparameters
dif (float): How much to perturb the hyperparameters

Returns:
vector: An estimate of the gradient

"""
function est_∇nlogL_GLOM(prob_def::GLO, total_hyperparameters::Vector{T}; dif::Real=1e-7) where {T<:Real}

    f(inputs) = nlogL_GLOM(prob_def, inputs)
    return est_∇(f, total_hyperparameters; dif=dif, ignore_0_inputs=true)

end


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
            println("mismatch df/dθ" * string(i))
            no_mismatch = false
        end
    end

    return no_mismatch
end

"""
Test that the analytical and numerically estimated ∇nlogL_GLOM() are approximately the same

Parameters:

prob_def (GLO): A structure that holds all of the relevant
    information for constructing the model used in the GLOM et al. 2017+ paper
kernel_hyperparameters (vector): The kernel hyperparameters for the GP model
dif (float): How much to perturb the hyperparameters
print_stuff (bool): if true, prints extra information about the output

Returns:
bool: Whether or not every part of the analytical and numerical gradients match

"""
function test_∇nlogL_GLOM(prob_def::GLO, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    G = ∇nlogL_GLOM(prob_def, total_hyperparameters)
    est_G = est_∇nlogL_GLOM(prob_def, total_hyperparameters; dif=dif)

    return test_∇(est_G, G; print_stuff=print_stuff, function_name="∇nlogL_GLOM")

end


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
Estimate the Hessian of nlogL_GLOM() with forward differences

Parameters:

prob_def (GLO): A structure that holds all of the relevant
    information for constructing the model used in the GLOM et al. 2017+ paper
total_hyperparameters (vector): The hyperparameters for the GP model, including
    both the coefficient hyperparameters and the kernel hyperparameters
dif (float): How much to perturb the hyperparameters

Returns:
matrix: An estimate of the hessian
est_∇∇nlogL_GLOM
"""
function est_∇∇nlogL_GLOM(prob_def::GLO, total_hyperparameters::Vector{T}; dif::Real=1e-4) where {T<:Real}

    g(inputs) = ∇nlogL_GLOM(prob_def, inputs)
    return est_∇∇(g, total_hyperparameters; ignore_0_inputs=true)

end


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

    if !no_mismatch
        println("mismatches at 0s")
        for i in 1:size(H, 1)
            println(matches[i, :])
        end
    end

    return no_mismatch
end


"""
Test that the analytical and numerically estimated ∇∇nlogL_GLOM() are approximately the same

Parameters:

prob_def (GLO): A structure that holds all of the relevant
    information for constructing the model used in the GLOM et al. 2017+ paper
kernel_hyperparameters (vector): The kernel hyperparameters for the GP model
dif (float): How much to perturb the hyperparameters
print_stuff (bool): if true, prints extra information about the output

Returns:
bool: Whether or not every part of the analytical and numerical Hessians match

"""
function test_∇∇nlogL_GLOM(prob_def::GLO, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    H = ∇∇nlogL_GLOM(prob_def, total_hyperparameters)
    est_H = est_∇∇nlogL_GLOM(prob_def, total_hyperparameters; dif=dif)

    return test_∇∇(est_H, H; print_stuff=print_stuff, function_name="∇∇nlogL_GLOM")

end
