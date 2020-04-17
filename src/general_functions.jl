"if array is symmetric, return the symmetric (and optionally cholesky factorized) version"
function symmetric_A(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}; ignore_asymmetry::Bool=false, chol::Bool=false) where {T<:Real}

    # an arbitrary threshold that is meant to catch numerical errors
    thres = maximum([1e-6 * maximum(abs.(A)), 1e-8])

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            A = Symmetric(A)

        elseif (max_dif < thres) || ignore_asymmetry
            # return the symmetrized version of the matrix
            A = symmetrize_A(A)
        else
            println("Array dimensions match, but the max dif ($max_dif) is greater than the threshold ($thres)")
            chol = false
        end
    else
        println("Array dimensions do not match. The matrix can't be symmetric")
        chol = false
    end

    return chol ? ridge_chol(A) : A

end


function symmetrize_A(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}) where {T<:Real}
    return Symmetric((A + transpose(A)) / 2)
end


"if needed, adds a ridge based on the smallest eignevalue to make a Cholesky factorization possible"
function ridge_chol(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}) where {T<:Real}

    # only add a small ridge (based on the smallest eigenvalue) if necessary
    try
        return cholesky(A)
    catch
        smallest_eigen = IterativeSolvers.lobpcg(A, false, 1).λ[1]
        ridge = 1.10 * abs(smallest_eigen)
        @warn "added a ridge"
        println("ridge size:          10^$(log10(ridge))")
        println("max value of array:  10^$(log10(maximum(abs.(A))))")
        return cholesky(A + UniformScaling(ridge))
    end

end

"dont do anything if an array that is already factorized is passed"
ridge_chol(A::Cholesky{T,Matrix{T}}) where {T<:Real} = A


"Create a new array filling the non-zero entries of a template array with a vector of values"
function reconstruct_array(non_zero_entries, template_array::Matrix{T}) where {T<:Real}
    @assert length(findall(!iszero, template_array))==length(non_zero_entries)
    new_array = zeros(size(template_array))
    new_array[findall(!iszero, template_array)] = non_zero_entries
    return new_array
end


"assert all passed variables are positive"
function assert_positive(vars...)
    for i in vars
        @assert all(ustrip.(i) .> 0) "passed a negative/0 variable that needs to be positive"
    end
end


ndims(A::Cholesky{T,Matrix{T}}) where {T<:Real} = 2


"""
For distributed computing. Send a variable to a worker
stolen shamelessly from ParallelDataTransfer.jl
e.g.
sendto([1, 2], x=100, y=rand(2, 3))
z = randn(10, 10); sendto(workers(), z=z)
"""
function sendto(workers::Union{T,Vector{T}}; args...) where {T<:Integer}
    for worker in workers
        for (var_name, var_value) in args
            @spawnat(worker, Core.eval(Main, Expr(:(=), var_name, var_value)))
        end
    end
end


"""
Automatically adds as many workers as there are CPU threads minus 2 if none are
active and no number of procs to add is given
"""
function auto_addprocs(;add_procs::Integer=0)
    # only add as any processors as possible if we are on a consumer chip
    if (add_procs==0) && (nworkers()==1)
        add_procs = length(Sys.cpu_info()) - 2
    end
    addprocs(add_procs)
    println("added $add_procs workers")
end


"finds -1 ^ power without calling ^"
powers_of_negative_one(power::Integer) = iseven(power) ? 1 : -1


"Return the a version of the passed vector after removing all zero entries"
remove_zeros(V::Vector{T} where T<:Real) = V[findall(!iszero, V)]


"""
Compute the logarithm of the Laplace approxmation for the integral of a function
of the following form
∫ exp(-λ g(y)) h(y) dy ≈ exp(-λ g(y*)) h(y*) (2π/λ)^(d/2) |H(y*)|^(-1/2)
where y* is the value of y at the global mode and H is the (Hessian) matrix of
second order partial derivatives of g(y) (see slide 10 of
http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/laplace.pdf). When used to
calculate evidences, one can set λ = 1, g(y) = -log-likelihood,
h(y) = model prior, and H(y) = the Fisher information matrix (FIM) or the
Hessian matrix of the negative log-likelihood. Possible to improve with methods
from Ruli et al. 2016 (https://arxiv.org/pdf/1502.06440.pdf)?

Parameters:

H (matrix): Hessian matrix of second order partial derivatives of g(y) at y*
g (float): g(y*) in above formula
logh (float): log(h(y*)) in above formula
λ (float): λ in above formula

Returns:
float: An estimate of log(∫ exp(-λ g(y)) h(y) dy)

"""
function log_laplace_approximation(
    H::Union{Symmetric{T,Matrix{T}},Matrix{T}},
    g::Real,
    logh::Real;
    λ = 1
    ) where {T<:Real}

    @assert size(H, 1) == size(H, 2)
    n = size(H, 1)

    return logh - λ * g + 0.5 * (n * log(2 * π / λ) - logdet(H))

end
