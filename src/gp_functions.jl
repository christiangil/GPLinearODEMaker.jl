"""
    covariance!(Σ, kernel_func, x1list, x2list, kernel_hyperparameters; dorder=zeros(Int64, 2 + length(kernel_hyperparameters)), symmetric=false)

Fills `Σ` with a covariance matrix by evaluating `kernel_func` with
`kernel_hyperparameters`for each pair of `x1list` and `x2list` entries.

# Keyword Arguments
- `dorder::Vector{T2}=zeros(Int64, 2 + length(kernel_hyperparameters))`: How often to differentiate the covariance function w.r.t each `kernel_hyperparameter`. (i.e. dorder=[1, 0, 1] would correspond to differenting once w.r.t. the first and third `kernel_hyperparameter`)
- `symmetric::Bool=false`: If you know that the resulting covariance matrix should be symmetric, setting this to `true` can reduce redundant calculations
"""
function covariance!(
    Σ::DenseArray{T1,2},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1};
    dorder::Vector{T2}=zeros(Int64, 2 + length(kernel_hyperparameters)),
    symmetric::Bool=false
    ) where {T1<:Real, T2<:Integer}

    @assert issorted(x1list)

    # are the list of x's passed identical
    same_x = (x1list == x2list)

    # are the x's passed identical and equally spaced
    if same_x
        spacing = x1list[2:end]-x1list[1:end-1]
        equal_spacing = all((spacing .- spacing[1]) .< (1e-8 * spacing[1]))
    else
        @assert issorted(x2list)
        equal_spacing = false
    end

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ, 1) == x1_length
    @assert size(Σ, 2) == x2_length

    if equal_spacing && symmetric
        # this section is so fast, it isn't worth parallelizing
        kernline = zeros(x1_length)
        for i in 1:x1_length
            kernline[i] = kernel_func(kernel_hyperparameters, x1list[1] - x1list[i], dorder)
        end
        for i in 1:x1_length
            Σ[i, i:end] = kernline[1:(x1_length + 1 - i)]
        end
        return Symmetric(Σ)
    else
        _covariance!(Σ, kernel_func, x1list, x2list, kernel_hyperparameters, dorder, symmetric, same_x, equal_spacing)
    end
end

"""
    _covariance!(Σ::SharedArray, kernel_func, x1list, x2list, kernel_hyperparameters, dorder, symmetric, same_x, equal_spacing)

Fills `Σ` in parallel with a covariance matrix by evaluating `kernel_func` with
`kernel_hyperparameters`for each pair of `x1list` and `x2list` entries. Be
careful using a function that starts with a `_`.
"""
function _covariance!(
    Σ::SharedArray{T1,2},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1},
    dorder::Vector{T2},
    symmetric::Bool,
    same_x::Bool,
    equal_spacing::Bool
    ) where {T1<:Real, T2<:Integer}

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ) == (x1_length, x2_length)

    if same_x && symmetric
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, dorder=dorder)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x1list)
                if i <= j; Σ[i, j] = kernel_func(kernel_hyperparameters, x1list[i] - x1list[j], dorder) end
            end
        end
        return Symmetric(Σ)
    else
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, x2list=x2list, dorder=dorder)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x2list)
                Σ[i, j] = kernel_func(kernel_hyperparameters, x1list[i] - x2list[j], dorder)
            end
        end
        return Σ
    end
end

"""
_covariance!(Σ::Matrix, kernel_func, x1list, x2list, kernel_hyperparameters, dorder, symmetric, same_x, equal_spacing)

Fills `Σ` serially with a covariance matrix by evaluating `kernel_func` with
`kernel_hyperparameters`for each pair of `x1list` and `x2list` entries. Be
careful using a function that starts with a `_`.
"""
function _covariance!(
    Σ::Matrix{T1},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1},
    dorder::Vector{T2},
    symmetric::Bool,
    same_x::Bool,
    equal_spacing::Bool
    ) where {T1<:Real, T2<:Integer}

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ) == (x1_length, x2_length)

    if same_x && symmetric
        for i in 1:length(x1list)
            for j in 1:length(x1list)
                if i <= j; Σ[i, j] = kernel_func(kernel_hyperparameters, x1list[i] - x1list[j], dorder) end
            end
        end
        return Symmetric(Σ)
    else
        for i in 1:length(x1list)
            for j in 1:length(x2list)
                Σ[i, j] = kernel_func(kernel_hyperparameters, x1list[i] - x2list[j], dorder)
            end
        end
        return Σ
    end
end

"""
    covariance(kernel_func, x1list, x2list, kernel_hyperparameters; kwargs...)

Calculates a covariance matrix by evaluating `kernel_func` with
`kernel_hyperparameters`for each pair of `x1list` and `x2list` entries.

See also: [`covariance!`](@ref)
"""
function covariance(
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1};
    kwargs...) where {T1<:Real}

    if nworkers()>1
        return covariance!(SharedArray{Float64}(length(x1list), length(x2list)), kernel_func, x1list, x2list, kernel_hyperparameters; kwargs...)
    else
        return covariance!(zeros(length(x1list), length(x2list)), kernel_func, x1list, x2list, kernel_hyperparameters; kwargs...)
    end
end


"""
    covariance(glo, x1list, x2list, total_hyperparameters; dΣdθs_total=Int64[], kwargs...)

Calculates the total GLOM covariance matrix by combining the latent covariance
matrices implied by `glo` for each pair of `x1list` and `x2list` entries.

# Notable Arguments
- `total_hyperparameters::Vector`: The current `a` values (GLOM coeffients that describe how to combine the differentiated versions of the latent GP) followed by the `kernel_hyperparameters` (i.e. lengthscales and periods)
- `dΣdθs_total=Int64[]`: Which of the `total_hyperparameters` to differentiate the covariance function w.r.t. (i.e. for a `glo` with 6 `a` values and 2 kernel hyperparameters, dΣdθs_total=[4, 8] would correspond to differenting once w.r.t. the fourth `a` value and second kernel hyperparameter)

See also: [`covariance!`](@ref)
"""
function covariance(
    glo::GLO,
    x1list::Vector{<:Real},
    x2list::Vector{<:Real},
    total_hyperparameters::Vector{<:Real};
    dΣdθs_total::Vector{<:Integer}=Int64[],
    kwargs...)

    @assert all(dΣdθs_total .>= 0)
    @assert length(total_hyperparameters) == glo.n_kern_hyper + length(glo.a)

    num_coefficients = length(total_hyperparameters) - glo.n_kern_hyper
    n_out = glo.n_out
    n_dif = glo.n_dif
    dΣdθs_kernel = dΣdθs_total .- num_coefficients
    kernel_hyperparameters = total_hyperparameters[(num_coefficients + 1):end]
    # println(length(kernel_hyperparameters))

    x1_length = length(x1list)
    x2_length = length(x2list)

    if nworkers()>1
        holder = SharedArray{Float64}(length(x1list), length(x2list))
    else
        holder = zeros(length(x1list), length(x2list))
    end

    # only calculating each sub-matrix once and using the fact that they should
    # be basically the same if the kernel has been differentiated the same amount of times
    A_list = Vector{AbstractArray{<:Real,2}}(undef, 2 * n_dif - 1)
    dorder = zeros(Int64, 2 + length(kernel_hyperparameters))
    for dΣdθ_kernel in dΣdθs_kernel
        if dΣdθ_kernel > 0; dorder[2 + dΣdθ_kernel] += 1 end
    end
    if !glo.kernel_changes_with_output
        for i in 0:(2 * n_dif - 2)
            dorder[1] = rem(i - 1, 2) + 1
            dorder[2] = 2 * div(i - 1, 2)
            # things that have been differentiated an even amount of times are symmetric about t1-t2==0
            if iseven(i)
                A_list[i + 1] = copy(covariance!(holder, glo.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=true))
            else
                A_list[i + 1] = copy(covariance!(holder, glo.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder))
            end
        end
    end
    # assembling the coefficient matrix
    a = reshape(total_hyperparameters[1:num_coefficients], (n_out, n_dif))

    # initializing the multi-output covariance matrix
    Σ = zeros((n_out * x1_length, n_out * x2_length))

    coeff_orders = copy(glo.coeff_orders)
    coeff_coeffs = copy(glo.coeff_coeffs)
    dif_coefficients!(n_out, n_dif, dΣdθs_total, coeff_orders, coeff_coeffs)
    for i in 1:n_out
        for j in 1:n_out
            # if i <= j  # only need to calculate one set of the off-diagonal blocks
                if glo.kernel_changes_with_output
                    outputs = [i,j]
                    for i in 0:(2 * n_dif - 2)
                        dorder[1] = rem(i - 1, 2) + 1
                        dorder[2] = 2 * div(i - 1, 2)
                        # things that have been differentiated an even amount of times are symmetric about t1-t2==0
                        if iseven(i) && outputs[1]==outputs[2]
                            A_list[i + 1] = copy(covariance!(holder, (hyper, δ, dord) -> glo.kernel(hyper, δ, dord; outputs=outputs), x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=true))
                        else
                            A_list[i + 1] = copy(covariance!(holder, (hyper, δ, dord) -> glo.kernel(hyper, δ, dord; outputs=outputs), x1list, x2list, kernel_hyperparameters; dorder=dorder))
                        end
                    end
                end
                for k in 1:n_dif
                    for l in 1:n_dif
                        # if the coefficient for the GLOM coefficients is non-zero
                        if coeff_coeffs[i, j, k, l] != 0
                            # make it negative or not based on how many times it has been differentiated in the x1 direction
                            A_mat_coeff = coeff_coeffs[i, j, k, l] * powers_of_negative_one(l - 1)
                            for m in 1:n_out
                                for n in 1:n_dif
                                    A_mat_coeff *= a[m, n] ^ coeff_orders[i, j, k, l, m, n]
                                end
                            end
                            # add the properly negative differentiated A matrix from the list
                            Σ[((i - 1) * x1_length + 1):(i * x1_length),
                                ((j - 1) * x2_length + 1):(j * x2_length)] +=
                                A_mat_coeff * A_list[k + l - 1]
                        end
                    end
                end
            # end
        end
    end

    # making the highest variances all along the main diagonals
    #TODO implement in above step (aka at initial assignment)
    Σ_rearranged = zeros((n_out * x1_length, n_out * x2_length))
    for i in 1:x1_length
        for j in 1:x2_length
            Σ_rearranged[1 + (i - 1) * n_out:i * n_out,1 + (j - 1) * n_out:j * n_out] = Σ[i:x1_length:end,:][:, j:x2_length:end]
        end
    end
    # return the symmetrized version of the covariance matrix
    # function corrects for numerical errors and notifies us if our matrix isn't
    # symmetric like it should be
    return symmetric_A(Σ_rearranged; kwargs...)
    # if x1list == x2list
    #     if chol
    #         return ridge_chol(Symmetric(Σ))
    #     else
    #         return Symmetric(Σ)
    #     end
    # else
    #     return Σ
    # end

end
"""
    covariance(glo, total_hyperparameters; kwargs...)

Calculates the total GLOM covariance matrix by combining the latent covariance
matrices implied by `glo` at each `glo.x_obs`

See also: [`covariance`](@ref)
"""
covariance(
    glo::GLO,
    total_hyperparameters::Vector{<:Real};
    kwargs...
    ) = covariance(glo, glo.x_obs, glo.x_obs, total_hyperparameters; kwargs...)


"""
    Σ_observations(Σ, measurement_noise::Vector; return_both=false, kwargs...)

Add `measurement_noise`^2 to the diagonal of `Σ` and performs a Cholesky
factorization. Optionally returns the `Σ` back as well.
"""
function Σ_observations(
    Σ::Symmetric{T, Matrix{T}},
    measurement_noise::Vector{T};
    return_both::Bool=false,
    kwargs...) where {T<:Real}

    Σ_obs = symmetric_A(Σ + Diagonal(measurement_noise .* measurement_noise); chol=true, kwargs...)
    return return_both ? (Σ_obs, Σ) : Σ_obs
end

"""
    Σ_observations(Σ, measurement_covariance::Array{T, 3}; return_both=false, kwargs...)

Add `measurement_covariance` to the block diagonal of `Σ` and performs a
Cholesky factorization. Optionally returns the `Σ` back as well.
"""
function Σ_observations(
    Σ::Symmetric{T, Matrix{T}},
    measurement_covariance::Array{T, 3};
    return_both::Bool=false,
    kwargs...) where {T<:Real}

    n_meas = size(measurement_covariance, 1)
    n_out = size(measurement_covariance, 2)
    @assert n_out == size(measurement_covariance, 3) == size(Σ, 2) / n_meas
    Σ_obs = Matrix(Σ)
    # for i in 1:n_out
    #     for j in 1:n_out
    #         if i <= j; Σ_obs[diagind(Σ_obs, i - 1)[((j - i) * n_meas + 1):((j - i + 1) * n_meas)]] += measurement_covariance[:, i, j] end
    #     end
    # end
    for i in 1:n_meas
        Σ_obs[1 + (i - 1) * n_out : i * n_out, 1 + (i - 1) * n_out : i * n_out] += measurement_covariance[i, :, :]
    end
    Σ_obs = symmetric_A(Σ_obs; chol=true, kwargs...)
    return return_both ? (Σ_obs, Σ) : Σ_obs
end

"""
    Σ_observations(kernel_func, x_obs, measurement_noise, kernel_hyperparameters; ignore_asymmetry=false, return_both)

Calculates the covariance matrix of `kernel_func` at `x_obs` and adds
`measurement_noise`^2 to the diagonal and performs a Cholesky factorization.
Optionally returns the `Σ` back as well.
"""
Σ_observations(
    kernel_func::Function,
    x_obs::Vector{T},
    measurement_noise::Vector{T},
    kernel_hyperparameters::Vector{T};
    ignore_asymmetry::Bool=false,
    kwargs...
    ) where {T<:Real} =
    Σ_observations(
        symmetric_A(
            covariance(kernel_func, x_obs, x_obs, kernel_hyperparameters);
            ignore_asymmetry=ignore_asymmetry),
        measurement_noise;
        kwargs...)

"""
    Σ_observations(glo, total_hyperparameters; ignore_asymmetry=false, return_both=false)

Calculates a Cholesky decomposition of the GLOM covariance matrix implied by
`glo` and `total_hyperparameters` including `glo.noise` or `glo.covariance` on
the (block) diagonal
"""
function Σ_observations(
    glo::GLO,
    total_hyperparameters::Vector{T};
    ignore_asymmetry::Bool=false,
    kwargs...
    ) where {T<:Real}

    return glo.has_covariance ?
        Σ_observations(symmetric_A(covariance(glo, total_hyperparameters); ignore_asymmetry=ignore_asymmetry), glo.covariance; kwargs...) :
        Σ_observations(symmetric_A(covariance(glo, total_hyperparameters); ignore_asymmetry=ignore_asymmetry), glo.noise; kwargs...)
end


"""
    get_σ(L_obs, Σ_obs_samp, diag_Σ_samp)

Calculate the GP posterior standard deviation at each sampled point. Algorithm
2.1 from Rasmussen and Williams
"""
function get_σ(
    L_obs::LowerTriangular{T,Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric,Matrix{T}},
    diag_Σ_samp::Vector{T}
    ) where {T<:Real}

    v = L_obs \ Σ_obs_samp
    # thing = diag_Σ_samp - [dot(v[:, i], v[:, i]) for i in 1:length(diag_Σ_samp)]
    # thing[abs.(thing) .< 1e-10] .= 0
    # return sqrt.(thing)  # σ
    return sqrt.(diag_Σ_samp - [dot(v[:, i], v[:, i]) for i in 1:length(diag_Σ_samp)])  # σ
    # return sqrt.(diag_Σ_samp - diag(Σ_samp_obs * (Σ_obs \ Σ_obs_samp)))  # much slower
end

"""
    get_σ(glo, x_samp, total_hyperparameters)

Calculate the `glo` GP (using `total_hyperparameters`) posterior standard
deviation at each `x_samp` point.
"""
function get_σ(
    glo::GLO,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T}
    ) where {T<:Real}

    (Σ_samp, Σ_obs, _, Σ_obs_samp) = covariance_permutations(glo, x_samp, total_hyperparameters)
    return get_σ(ridge_chol(Σ_obs).L, Σ_obs_samp, diag(Σ_samp))
end

"""
    covariance_permutations(kernel_func, x_obs, x_samp, measurement_noise, kernel_hyperparameters; return_both=false)

Calculate all of the different versions of the covariance matrices using
`kernel_func` with `kernel_hyperparameters` between each of the pairs of `x_obs`
and `x_samp` and themselves.
"""
function covariance_permutations(
    kernel_func::Function,
    x_obs::Vector{T},
    x_samp::Vector{T},
    measurement_noise::Vector{T},
    kernel_hyperparameters::Vector{T};
    return_both = false,
    kwargs...
    ) where {T<:Real}

    Σ_samp = covariance(kernel_func, x_samp, x_samp, kernel_hyperparameters)
    Σ_samp_obs = covariance(kernel_func, x_samp, x_obs, kernel_hyperparameters; ignore_asymmetry=true)
    Σ_obs_samp = transpose(Σ_samp_obs)
    if return_both
        Σ_obs, Σ_obs_raw = Σ_observations(kernel_func, x_obs, measurement_noise, kernel_hyperparameters; return_both=return_both, kwargs...)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw
    else
        Σ_obs = Σ_observations(kernel_func, x_obs, measurement_noise, kernel_hyperparameters; return_both=return_both, kwargs...)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp
    end
end

"""
    covariance_permutations(glo, x_samp, total_hyperparameters; return_both=false)

Calculate all of the different versions of the covariance matrices using
the `glo` GP with `total_hyperparameters` between each of the pairs of
`glo.x_obs` and `x_samp` and themselves.
"""
function covariance_permutations(
    glo::GLO,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T};
    return_both = false,
    kwargs...
    ) where {T<:Real}

    Σ_samp = covariance(glo, x_samp, x_samp, total_hyperparameters)
    Σ_samp_obs = covariance(glo, x_samp, glo.x_obs, total_hyperparameters; ignore_asymmetry=true)
    Σ_obs_samp = transpose(Σ_samp_obs)
    if return_both
        Σ_obs, Σ_obs_raw = Σ_observations(glo, total_hyperparameters; return_both=return_both, kwargs...)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw
    else
        Σ_obs = Σ_observations(glo, total_hyperparameters; return_both=return_both, kwargs...)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp
    end
end


"""
    GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw; return_Σ=true, kwargs...)

Calculate the sampled posterior mean and std, observed posterior mean, and
(optionally) the posterior covariance matrix for the GP used to calculate
`Σ_samp`, `Σ_obs`, `Σ_samp_obs`, `Σ_obs_samp`, and `Σ_obs_raw`
"""
function GP_posteriors_from_covariances(
    y_obs::Vector{T},
    Σ_samp::Union{Cholesky,Symmetric,Matrix{T}},
    Σ_obs::Cholesky,
    Σ_samp_obs::Union{Symmetric,Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric,Matrix{T}},
    Σ_obs_raw::Symmetric;
    return_Σ::Bool=true,
    kwargs...
    ) where {T<:Real}

    # posterior mean calcuation from RW alg. 2.1

    # these are all equivalent but have different computational costs
    # α = inv(Σ_obs) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = Σ_obs \ y_obs

    mean_post = Σ_samp_obs * α
    mean_post_obs = Σ_obs_raw * α

    # posterior standard deviation calcuation from RW alg. 2.1
    σ = get_σ(Σ_obs.L, Σ_obs_samp, diag(Σ_samp))

    # posterior covariance calculation is from eq. 2.24 of RW
    if return_Σ
        Σ_post = symmetric_A(Σ_samp - (Σ_samp_obs * (Σ_obs \ Σ_obs_samp)); kwargs...)
        return mean_post, σ, mean_post_obs, Σ_post
    else
        return mean_post, σ, mean_post_obs
    end

end


"""
    GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp; return_Σ=true, kwargs...)

Calculate the sampled posterior mean and std and (optionally) the posterior
covariance matrix for the GP used to calculate `Σ_samp`, `Σ_obs`, `Σ_samp_obs`,
and `Σ_obs_samp`
"""
function GP_posteriors_from_covariances(
    y_obs::Vector{T},
    Σ_samp::Union{Cholesky,Symmetric,Matrix{T}},
    Σ_obs::Cholesky,
    Σ_samp_obs::Union{Symmetric,Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric,Matrix{T}};
    return_Σ::Bool=true,
    kwargs...
    ) where {T<:Real}

    # posterior mean calcuation from RW alg. 2.1

    # these are all equivalent but have different computational costs
    # α = inv(Σ_obs) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = Σ_obs \ y_obs

    mean_post = Σ_samp_obs * α

    # posterior standard deviation calcuation from RW alg. 2.1
    σ = get_σ(Σ_obs.L, Σ_obs_samp, diag(Σ_samp))

    # posterior covariance calculation is from eq. 2.24 of RW
    if return_Σ
        Σ_post = symmetric_A(Σ_samp - (Σ_samp_obs * (Σ_obs \ Σ_obs_samp)); kwargs...)
        return mean_post, σ, Σ_post
    else
        return mean_post, σ
    end

end


"""
    GP_posteriors_from_covariances(y_obs, Σ_obs, Σ_obs_raw)

Calculate the observed posterior mean for the GP used to calculate `Σ_obs` and
`Σ_obs_raw`
"""
GP_posteriors_from_covariances(
    y_obs::Vector{T},
    Σ_obs::Cholesky,
    Σ_obs_raw::Symmetric) where {T<:Real} =
    Σ_obs_raw * (Σ_obs \ y_obs)


"""
    GP_posteriors(kernel_func, x_obs, y_obs, x_samp, measurement_noise, kernel_hyperparameters; return_mean_obs=false, kwargs...)

Calculate the posterior mean and std at `x_samp`, (optionally) posterior mean
at `x_obs` and (optionally) the posterior covariance matrix for the GP described
by `kernel_func` and `kernel_hyperparameters`
"""
function GP_posteriors(
    kernel_func::Function,
    x_obs::Vector{T},
    y_obs::Vector{T},
    x_samp::Vector{T},
    measurement_noise::Vector{T},
    kernel_hyperparameters::Vector{T};
    return_mean_obs::Bool=false,
    kwargs...) where {T<:Real}

    if return_mean_obs
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw) = covariance_permutations(kernel_func, x_obs, x_samp, measurement_noise, kernel_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw; kwargs...)
    else
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp) = covariance_permutations(kernel_func, x_obs, x_samp, measurement_noise, kernel_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp; kwargs...)
    end
end


"""
    GP_posteriors(glo, x_samp, total_hyperparameters; return_mean_obs=false, y_obs=glo.y_obs, kwargs...)

Calculate the posterior mean and std at `x_samp`, (optionally) posterior mean
at `glo.x_obs` and (optionally) the posterior covariance matrix for the GP described
by `glo` and `total_hyperparameters`
"""
function GP_posteriors(
    glo::GLO,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T};
    return_mean_obs::Bool=false,
    y_obs::Vector{T}=glo.y_obs,
    kwargs...) where {T<:Real}

    if return_mean_obs
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw) = covariance_permutations(glo, x_samp, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw; kwargs...)
    else
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp) = covariance_permutations(glo, x_samp, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp; kwargs...)
    end

end


"""
    GP_posteriors(glo, total_hyperparameters; y_obs=glo.y_obs, kwargs...)

Calculate the posterior mean at `glo.x_obs` for the GP described by `glo` and
`total_hyperparameters`
"""
function GP_posteriors(
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=glo.y_obs,
    kwargs...
    ) where {T<:Real}
    Σ_obs, Σ_obs_raw = Σ_observations(glo, total_hyperparameters; return_both=true, kwargs...)
    return GP_posteriors_from_covariances(y_obs, Σ_obs, Σ_obs_raw)
end


"""
    coefficient_orders(n_out, n_dif; a=ones(n_out, n_dif))

Find the powers that each GLOM coefficient is taken to for each part of the
matrix construction before differentiating by any hyperparameters.

# Outputs
- `coeff_orders::Array{Int, 6}`: filled with integers for what power each coefficient is taken to in the construction of a given block of the total covariance matrix. For example, coeff_orders[1,1,2,3,:,:] would tell you the powers of each coefficient (:,:) that are multiplied by the covariance matrix constructed by evaluating the partial derivative of the kernel (once by t1 and twice by t2) at every pair of time points (2,3) in the construction of the first block of the total covariance matrix (1,1)
- `coeff_coeffs::Array{Int, 4}`: Filled with ones anywhere that coeff_orders indicates that coefficients exists to multiply a given covariance matrix for a given block
"""
function coefficient_orders(
    n_out::Integer,
    n_dif::Integer;
    a::Matrix{T}=ones(n_out, n_dif)
    ) where {T<:Real}

    @assert size(a) == (n_out, n_dif)

    # ((output pair), (which A matrix to use), (which a coefficent to use))
    coeff_orders = zeros(Int64, n_out, n_out, n_dif, n_dif, n_out, n_dif)
    coeff_coeffs = zeros(Int64, n_out, n_out, n_dif, n_dif)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    for m in 1:n_out
                        for n in 1:n_dif
                            if a[m, n] != 0
                                if [m, n] == [i, k]; coeff_orders[i, j, k, l, m, n] += 1 end
                                if [m, n] == [j, l]; coeff_orders[i, j, k, l, m, n] += 1 end
                            end
                        end
                    end
                    # There should be two coefficients being applied to every
                    # matrix. If there are less than two, that means one of the
                    # coefficients was zero, so we should just set them both to
                    # zero
                    if sum(coeff_orders[i, j, k, l, :, :]) != 2
                        coeff_orders[i, j, k, l, :, :] .= 0
                    else
                        coeff_coeffs[i, j, k, l] = 1
                    end
                end
            end
        end
    end

    return coeff_orders, coeff_coeffs

end

"""
    dif_coefficients!(n_out, n_dif, dΣdθ_total::Int, coeff_orders, coeff_coeffs)

Modify `coeff_orders` and `coeff_coeffs` with the coefficients for constructing
differentiated version of the kernel (for the differentiation implied by
`dΣdθ_total`) using the powers that each coefficient is taken to for each part
of the matrix construction
"""
function dif_coefficients!(
    n_out::Integer,
    n_dif::Integer,
    dΣdθ_total::Integer,
    coeff_orders::AbstractArray{T,6},
    coeff_coeffs::AbstractArray{T,4}
    ) where {T<:Integer}

    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)

    # only do something if a derivative is being taken
    if dΣdθ_total > 0 && dΣdθ_total <= (n_out*n_dif)
        proper_indices = [((dΣdθ_total - 1) % n_out) + 1, div(dΣdθ_total - 1, n_out) + 1]
        for i in 1:n_out
            for j in 1:n_out
                for k in 1:n_dif
                    for l in 1:n_dif
                        if coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] != 0
                            coeff_coeffs[i, j, k, l] *= coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]]
                            coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] -= 1
                        else
                            coeff_coeffs[i, j, k, l] = 0
                            coeff_orders[i, j, k, l, :, :] .= 0
                        end
                    end
                end
            end
        end
    end
end


"""
    dif_coefficients!(n_out, n_dif, dΣdθs_total::Vector, coeff_orders, coeff_coeffs)

Modify `coeff_orders` and `coeff_coeffs` with the coefficients for constructing
differentiated version of the kernel (for the differentiations implied by
`dΣdθ_totals`) using the powers that each coefficient is taken to for each part
of the matrix construction
"""
function dif_coefficients!(
    n_out::Integer,
    n_dif::Integer,
    dΣdθs_total::Vector{T},
    coeff_orders::AbstractArray{T,6},
    coeff_coeffs::AbstractArray{T,4}
    ) where {T<:Integer}

    for dΣdθ_total in dΣdθs_total
        dif_coefficients!(n_out, n_dif, dΣdθ_total, coeff_orders, coeff_coeffs)
    end
end


# TODO: rename nlogL_normalization
"""
    nlogL(Σ, y; α= Σ \\ y, nlogL_normalization=logdet(Σ)+length(y)*log(2*π))

Negative log likelihood for data `y` assuming it was drawn from a multivariate
normal distribution with 0 mean and covariance `Σ` (usually `Σ + noise`)
"""
function nlogL(
    Σ::Union{Cholesky,Diagonal},
    y::Vector{T};
    α::Vector{T} = Σ \ y,
    nlogL_normalization::T=logdet(Σ)+length(y)*log(2*π)
    ) where {T<:Real}

    # n = length(y)

    # 2 times negative goodness of fit term
    data_fit = transpose(y) * α
    # 2 times negative complexity penalization term
    # complexity_penalty = log(det(Σ_obs))
    # complexity_penalty = logdet(Σ_obs)  # half memory but twice the time
    # 2 times negative normalization term (doesn't affect fitting)
    # normalization = n * log(2 * π)

    return (data_fit + nlogL_normalization) / 2

end


"""
    dnlogLdθ(y, α, β)

First partial derivative of `nlogL(Σ, y)` w.r.t. hyperparameters that affect `Σ`

# Arguments
- `y::Vector`: The observations at each time point
- `α::Vector`: `inv(Σ) * y`
- `β::Matrix`: `inv(Σ) * dΣ_dθ` where `dΣ_dθ` is the partial derivative of the `Σ` w.r.t. a hyperparameter
"""
function dnlogLdθ(
    y::Vector{T},
    α::Vector{T},
    β::Matrix{T}
    ) where {T<:Real}

    # 2 times negative derivative of goodness of fit term
    data_fit = -(transpose(y) * β * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β)

    # return -1 / 2 * tr((α * transpose(α) - inv(Σ_obs)) * dΣ_dθj)
    return (data_fit + complexity_penalty) / 2

end


"""
    dnlogLdθ(y1, α)

First partial derivative of `nlogL(Σ, y)` w.r.t. hyperparameters that affect `y`

# Arguments
- `y1::Vector`: The derivative of observations at each time point
- `α::Vector`: `inv(Σ) * y`
"""
function dnlogLdθ(
    y1::Vector{T},
    α::Vector{T}
    ) where {T<:Real}

    return transpose(y1) * α
    # return transpose(y) * α1
    # return (transpose(y1) * Σ^-1 * y + transpose(y) * Σ^-1 * y1) / 2
end


"""
    ∇nlogL(y, α, βs)

Gradient of `nlogL(Σ, y)` w.r.t. hyperparameters that affect `Σ`

# Arguments
- `y::Vector`: The observations at each time point
- `α::Vector`: inv(Σ) * y
- `βs::Vector{Matrix}`: List of `inv(Σ) * dΣ_dθ` where `dΣ_dθ` is the partial derivative of `Σ` w.r.t. each hyperparameter
"""
∇nlogL(y::Vector{T}, α::Vector{T}, βs::Vector{Matrix{T}}) where {T<:Real} = [dnlogLdθ(y, α, β) for β in βs]

"""
    d2nlogLdθ(y, α, β1, β2, β12)

Second partial derivative of `nlogL(Σ, y)` w.r.t. two hyperparameters that
affect `Σ`

# Arguments
- `y::Vector`: The observations at each time point
- `α::Vector`: `inv(Σ) * y`
- `β1::Matrix`: `inv(Σ) * dΣ_dθ` where `dΣ_dθ` is the partial derivative of the `Σ` w.r.t. the first hyperparameter
- `β2::Matrix`: Same as `β1` but for the second hyperparameter
- `β12::Matrix`: `inv(Σ_obs) * d2Σ_dθ1dθ2` where `d2Σ_dθ1dθ2` is the partial derivative of the covariance matrix Σ_obs w.r.t. both of the hyperparameters being considered
"""
function d2nlogLdθ(
    y::Vector{T},
    α::Vector{T},
    β1::Matrix{T},
    β2::Matrix{T},
    β12::Matrix{T}
    ) where {T<:Real}

    β12mβ2β1 = β12 - β2 * β1

    # 2 times negative second derivative of goodness of fit term
    data_fit = -(transpose(y) * (β12mβ2β1 - β1 * β2) * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β12mβ2β1)

    return (data_fit + complexity_penalty) / 2

end

"""
    d2nlogLdθ(y2, y12, α, α1)

Second partial derivative of `nlogL(Σ, y)` w.r.t. two parameters that affect `y`

# Arguments
- `y2::Vector`: The derivative of observations w.r.t the second parameter at each time point
- `y12::Vector`: The derivative of observations w.r.t the both parameters at each time point
- `α::Vector`: `inv(Σ) * y`
- `α1::Vector`: `inv(Σ) * y1` where `y1` is the derivative of observations w.r.t the first parameter at each time point
"""
function d2nlogLdθ(
    y2::Vector{T},
    y12::Vector{T},
    α::Vector{T},
    α1::Vector{T}
    ) where {T<:Real}

    return transpose(y12) * α + transpose(y2) * α1
    # return (transpose(y12) * α + transpose(y1) * α2 + transpose(y2) * α1 + transpose(y) * α12) / 2

end

"""
    d2nlogLdθ(y, y1, α, α1, β2)

Second partial derivative of `nlogL(Σ, y)` w.r.t. a parameter that affects `y` and a hyperparameter that affects `Σ`

# Arguments
- `y::Vector`: The observations at each time point
- `y1::Vector`: The derivative of observations w.r.t the `y`-affecting parameter at each time point
- `α::Vector`: `inv(Σ) * y`
- `α1::Vector`: `inv(Σ) * y1` where `y1` is the derivative of observations w.r.t the first parameter at each time point
- `β2::Matrix`: `inv(Σ) * dΣ_dθ` where `dΣ_dθ` is the partial derivative of the `Σ` w.r.t. the `Σ`-affecting hyperparameter
"""
function d2nlogLdθ(
    y::Vector{T},
    y1::Vector{T},
    α::Vector{T},
    α1::Vector{T},
    β2::Matrix{T}
    ) where {T<:Real}

    return -(transpose(y1) * β2 * α + transpose(y) * β2 * α1) / 2

end


"""
	nlogL_matrix_workspace{T<:Real}

A structure that holds all of the relevant information for calculating nlogL()
derivatives. Used to prevent recalculations during optimization.

# Arguments
- `nlogL_hyperparameters::Vector`: The current hyperparameters
- `Σ_obs::Cholesky`: The covariance matrix based on `nlogL_hyperparameters`
- `∇nlogL_hyperparameters::Vector`: The `nlogl` gradient at `nlogL_hyperparameters`
- `βs::Vector{Matrix}`: List of `inv(Σ) * dΣ_dθ` where `dΣ_dθ` is the partial derivative of `Σ` w.r.t. each `nlogL_hyperparameters`
"""
struct nlogL_matrix_workspace{T<:Real}
    nlogL_hyperparameters::Vector{T}
    Σ_obs::Cholesky
    ∇nlogL_hyperparameters::Vector{T}
    βs::Vector{Matrix{T}}

    function nlogL_matrix_workspace(glo::GLO,
        total_hyperparameters::Vector{<:Real})

        total_hyper = reconstruct_total_hyperparameters(glo::GLO, total_hyperparameters)
        Σ_obs = Σ_observations(glo, total_hyper; ignore_asymmetry=true)
        return nlogL_matrix_workspace(
            total_hyper,
            Σ_obs,
            copy(total_hyper),
            [Σ_obs \ covariance(glo, total_hyper; dΣdθs_total=[i]) for i in glo.non_zero_hyper_inds])
    end
    nlogL_matrix_workspace(
        nlogL_hyperparameters::Vector{T},
        Σ_obs::Cholesky,
        ∇nlogL_hyperparameters::Vector{T},
        βs::Vector{Matrix{T}}
        ) where T<:Real = new{typeof(nlogL_hyperparameters[1])}(nlogL_hyperparameters, Σ_obs, ∇nlogL_hyperparameters, βs)
end


"""
    calculate_shared_nlogL_matrices(glo, non_zero_hyperparameters; Σ_obs=Σ_observations(glo, reconstruct_total_hyperparameters(glo, non_zero_hyperparameters); ignore_asymmetry=true))

Calculates the quantities shared by the nlogL and ∇nlogL calculations
"""
function calculate_shared_nlogL_matrices(
    glo::GLO,
    non_zero_hyperparameters::Vector{<:Real};
    Σ_obs::Cholesky where T<:Real=Σ_observations(glo, reconstruct_total_hyperparameters(glo, non_zero_hyperparameters); ignore_asymmetry=true))

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(glo, non_zero_hyperparameters)

    return total_hyperparameters, Σ_obs

end


"""
    calculate_shared_nlogL_matrices!(workspace, glo, non_zero_hyperparameters)

Calculates the quantities shared by the nlogL and ∇nlogL calculations and stores
them in the existing `workspace`
"""
function calculate_shared_nlogL_matrices!(
    workspace::nlogL_matrix_workspace,
    glo::GLO,
    non_zero_hyperparameters::Vector{<:Real})

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(glo, non_zero_hyperparameters)

    if workspace.nlogL_hyperparameters != total_hyperparameters
        workspace.nlogL_hyperparameters[:] = total_hyperparameters
        workspace.Σ_obs.factors[:,:] = Σ_observations(glo, total_hyperparameters).factors
    end

end


"""
    calculate_shared_∇nlogL_matrices(glo, non_zero_hyperparameters; kwargs...)

Calculates the quantities shared by the ∇nlogL and ∇∇nlogL calculations
"""
function calculate_shared_∇nlogL_matrices(
    glo::GLO,
    non_zero_hyperparameters::Vector{<:Real};
    kwargs...)

    total_hyperparameters, Σ_obs = calculate_shared_nlogL_matrices(glo, non_zero_hyperparameters; kwargs...)

    βs = [Σ_obs \ covariance(glo, total_hyperparameters; dΣdθs_total=[i]) for i in glo.non_zero_hyper_inds]

    return total_hyperparameters, Σ_obs, βs

end


"""
    calculate_shared_∇nlogL_matrices!(workspace, glo, non_zero_hyperparameters)

Calculates the quantities shared by the ∇nlogL and ∇∇nlogL calculations and
stores them in the existing `workspace`
"""
function calculate_shared_∇nlogL_matrices!(
    workspace::nlogL_matrix_workspace,
    glo::GLO,
    non_zero_hyperparameters::Vector{<:Real})

    calculate_shared_nlogL_matrices!(workspace, glo, non_zero_hyperparameters)

    if workspace.∇nlogL_hyperparameters != workspace.nlogL_hyperparameters
        workspace.∇nlogL_hyperparameters[:] = workspace.nlogL_hyperparameters
        workspace.βs[:] = [workspace.Σ_obs \ covariance(glo, workspace.∇nlogL_hyperparameters; dΣdθs_total=[i]) for i in findall(!iszero, workspace.∇nlogL_hyperparameters)]
    end

end


"""
    include_kernel(kernel_name)

Tries to include the specified kernel, assuming it was included with GLOM.
Returns the kernel function and number of hyperparameters it uses
"""
function include_kernel(kernel_name::AbstractString)
    if !occursin("_kernel", kernel_name)
        kernel_name *= "_kernel"
    end
    return include(joinpath(pathof(GPLinearODEMaker),"..","kernels", kernel_name * ".jl"))
end


"""
    prep_parallel_covariance(kernel_name, kernel_path; kwargs...)

Make it easy to run the covariance calculations on many processors. Makes sure
every worker has access to kernel function, importing it from the
`kernel_path`.
"""
function prep_parallel_covariance(
    kernel_name::AbstractString,
    kernel_path::AbstractString;
    kwargs...)

    prep_parallel(; kwargs...)
    sendto(workers(), kernel_name=kernel_name)
    @everywhere include(kernel_path)
end


"""
    prep_parallel_covariance(kernel_name; kwargs...)

Make it easy to run the covariance calculations on many processors. Makes sure
every worker has access to kernel function.
"""
function prep_parallel_covariance(
    kernel_name::AbstractString;
    kwargs...)

    prep_parallel(; kwargs...)
    sendto(workers(), kernel_name=kernel_name)
    @everywhere include_kernel(kernel_name)
end


"""
    nlogL_GLOM(glo, total_hyperparameters; y_obs=copy(glo.y_obs), kwargs...)

Negative log likelihood for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrix.
"""
function nlogL_GLOM(
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs),
    kwargs...) where {T<:Real}

    total_hyperparameters, Σ_obs = calculate_shared_nlogL_matrices(
        glo, remove_zeros(total_hyperparameters); kwargs...)

    return nlogL(Σ_obs, y_obs)
end


"""
    nlogL_GLOM!(workspace, glo, total_hyperparameters; y_obs=copy(glo.y_obs))

Negative log likelihood for `glo` using `total_hyperparameters` to construct the
covariance matrix and storing the intermediate results in `workspace`.
"""
function nlogL_GLOM!(
    workspace::nlogL_matrix_workspace,
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs)
    ) where {T<:Real}

    calculate_shared_nlogL_matrices!(workspace, glo, total_hyperparameters)
    return nlogL(workspace.Σ_obs, y_obs)
end


"""
    ∇nlogL_GLOM(glo, total_hyperparameters; y_obs=copy(glo.y_obs), kwargs...)

`nlogL` gradient for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrices.
"""
function ∇nlogL_GLOM(
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs),
    kwargs...) where {T<:Real}

    total_hyperparameters, Σ_obs, βs = calculate_shared_∇nlogL_matrices(
        glo, remove_zeros(total_hyperparameters); kwargs...)

    return ∇nlogL(y_obs, Σ_obs \ y_obs, βs)

end


"""
    ∇nlogL_GLOM!(workspace, glo, total_hyperparameters; y_obs=copy(glo.y_obs))

`nlogL` gradient for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrices and storing the intermediate results in `workspace`.
"""
function ∇nlogL_GLOM!(
    workspace::nlogL_matrix_workspace,
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs)
    ) where {T<:Real}

    calculate_shared_∇nlogL_matrices!(workspace, glo, total_hyperparameters)

    return ∇nlogL(y_obs, workspace.Σ_obs \ y_obs, workspace.βs)

end


"""
    ∇∇nlogL_GLOM(glo, total_hyperparameters, Σ_obs, y_obs, α, βs)

`nlogL` Hessian for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrices.
"""
function ∇∇nlogL_GLOM(
    glo::GLO,
    total_hyperparameters::Vector{T},
    Σ_obs::Cholesky,
    y_obs::Vector{T},
    α::Vector{T},
    βs::Array{Matrix{T},1}
    ) where {T<:Real}

    non_zero_inds = copy(glo.non_zero_hyper_inds)
    H = zeros(length(non_zero_inds), length(non_zero_inds))
    for (i, nzind1) in enumerate(non_zero_inds)
        for (j, nzind2) in enumerate(non_zero_inds)
            if i <= j
                H[i, j] = d2nlogLdθ(y_obs, α, βs[i], βs[j], Σ_obs \ covariance(glo, total_hyperparameters; dΣdθs_total=[nzind1, nzind2]))
            end
        end
    end
    H = Symmetric(H)
    return H

end


"""
    ∇∇nlogL_GLOM(glo, total_hyperparameters; y_obs=copy(glo.y_obs))

`nlogL` Hessian for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrices.
"""
function ∇∇nlogL_GLOM(
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs),
    kwargs...) where {T<:Real}

    total_hyperparameters, Σ_obs, βs = calculate_shared_∇nlogL_matrices(
        glo, remove_zeros(total_hyperparameters); kwargs...)

    return ∇∇nlogL_GLOM(glo, total_hyperparameters, Σ_obs, y_obs, Σ_obs \ y_obs, βs)

end


"""
    ∇∇nlogL_GLOM!(workspace, glo, total_hyperparameters; y_obs=copy(glo.y_obs))

`nlogL` Hessian for `glo` using the non-zero `total_hyperparameters` to
construct the covariance matrices and storing the intermediate results in
`workspace`.
"""
function ∇∇nlogL_GLOM!(
    workspace::nlogL_matrix_workspace,
    glo::GLO,
    total_hyperparameters::Vector{T};
    y_obs::Vector{T}=copy(glo.y_obs)
    ) where {T<:Real}

    calculate_shared_∇nlogL_matrices!(workspace, glo, total_hyperparameters)

    return ∇∇nlogL_GLOM(glo, workspace.nlogL_hyperparameters, workspace.Σ_obs, y_obs, workspace.Σ_obs \ y_obs, workspace.βs)

end


"""
    reconstruct_total_hyperparameters(glo, hyperparameters)

Reinsert the zero coefficients into the non-zero hyperparameter list if needed
"""
function reconstruct_total_hyperparameters(
    glo::GLO,
    hyperparameters::Vector{T}
    ) where {T<:Real}

    if length(hyperparameters)!=(glo.n_kern_hyper + length(glo.a))
        new_coeff_array = reconstruct_array(hyperparameters[1:end - glo.n_kern_hyper], glo.a)
        coefficient_hyperparameters = collect(Iterators.flatten(new_coeff_array))
        total_hyperparameters = append!(coefficient_hyperparameters, hyperparameters[end - glo.n_kern_hyper + 1:end])
    else
        total_hyperparameters = copy(hyperparameters)
    end

    @assert length(total_hyperparameters)==(glo.n_kern_hyper + length(glo.a))

    return total_hyperparameters

end
