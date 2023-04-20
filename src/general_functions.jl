"""
    symmetric_A(A; ignore_asymmetry=false, chol=false)

Check if `A` is approximately symmetric and then return a symmetrized version
or, optionally, the Cholesky factorization

# Examples
```jldoctest
julia> A = [1. 1; 0 1];

julia> GPLinearODEMaker.symmetric_A(A; ignore_asymmetry=true)
2×2 LinearAlgebra.Symmetric{Float64,Array{Float64,2}}:
 1.0  0.5
 0.5  1.0
```
"""
function symmetric_A(A::Union{Matrix{T},Symmetric}; ignore_asymmetry::Bool=false, chol::Bool=false) where {T<:Real}

    # an arbitrary threshold that is meant to catch numerical errors
    thres = max(1e-6 * maximum(abs.(A)), 1e-8)

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            A = Symmetric(A)

        elseif (max_dif < thres) || ignore_asymmetry
            # return the symmetrized version of the matrix
            A = symmetrize_A(A)
        else
            @warn "Array dimensions match, but the max dif ($max_dif) is greater than the threshold ($thres)"
            chol = false
        end
    else
        if !ignore_asymmetry
            @warn "Array dimensions do not match. The matrix can't be symmetric"
        end
        chol = false
    end

    return chol ? ridge_chol(A) : A

end


"""
    symmetrize_A(A)

Symmetrize `A` (i.e. add its transpose and divide by 2)

# Examples
```jldoctest
julia> A = [1. 1; 0 1];

julia> GPLinearODEMaker.symmetrize_A(A)
2×2 LinearAlgebra.Symmetric{Float64,Array{Float64,2}}:
 1.0  0.5
 0.5  1.0
```
"""
symmetrize_A(A::Union{Matrix{T},Symmetric}) where {T<:Real} =
    Symmetric((A + transpose(A)) / 2)
symmetrize_A(A::Symmetric) where {T<:Real} = A


"""
    ridge_chol(A)

Perform a Cholesky factorization on `A`, adding a small ridge when necessary

# Examples
```jldoctest
julia> A = [4. 2;2 10];

julia> GPLinearODEMaker.ridge_chol(A)
LinearAlgebra.Cholesky{Float64,Array{Float64,2}}
U factor:
2×2 LinearAlgebra.UpperTriangular{Float64,Array{Float64,2}}:
 2.0  1.0
  ⋅   3.0
```
"""
function ridge_chol(A::Union{Matrix{T},Symmetric}) where {T<:Real}

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
ridge_chol(A::Cholesky) where {T<:Real} = A

"""
    reconstruct_array(non_zero_entries, template_A)

Fill a matrix with `non_zero_entries` at the locations of the non-zero entries
in `template_A`

# Examples
```jldoctest
julia> template_A = [4. 0;5 0];

julia> non_zero_entries = [2., 3];

julia> GPLinearODEMaker.reconstruct_array(non_zero_entries, template_A)
2×2 Array{Float64,2}:
 2.0  0.0
 3.0  0.0
```
"""
function reconstruct_array(non_zero_entries, template_A::Matrix{T}) where {T<:Real}
    @assert length(findall(!iszero, template_A))==length(non_zero_entries)
    new_A = zeros(size(template_A))
    new_A[findall(!iszero, template_A)] = non_zero_entries
    return new_A
end


"""
    assert_positive(vars...)

# Examples
```jldoctest
julia> GPLinearODEMaker.assert_positive([1,2,3,4])

julia> GPLinearODEMaker.assert_positive([-1,2,3,4])
ERROR: AssertionError: passed a negative/0 variable that needs to be positive
[...]
```
"""
function assert_positive(vars...)
    for i in vars
        @assert all(i .> 0) "passed a negative/0 variable that needs to be positive"
    end
end


"""
    ndims(A::Cholesky)

Extends ndims to return 2 if passed a Cholesky object
"""
ndims(A::Cholesky) where {T<:Real} = 2


"""
    sendto(workers; args...)

Send the `args` to the `workers` specified by thier number IDs.
Stolen shamelessly from ParallelDataTransfer.jl

# Examples
```julia-repl
julia> sendto([1, 2], x=100, y=rand(2, 3));

julia> z = randn(10, 10); sendto(workers(), z=z);
```
"""
function sendto(workers::Union{T,Vector{T}}; args...) where {T<:Integer}
    for worker in workers
        for (var_name, var_value) in args
            @spawnat(worker, Core.eval(Main, Expr(:(=), var_name, var_value)))
        end
    end
end


"""
    auto_addprocs(;add_procs=0)

Adds either as many workers as there are CPU threads minus 2 if none are active
or the amount specified by `add_procs`
"""
function auto_addprocs(;add_procs::Integer=0)
    # only add as any processors as possible if we are on a consumer chip
    if (add_procs==0) && (nworkers()==1)
        add_procs = length(Sys.cpu_info()) - 2
    end
    addprocs(add_procs)
    println("added $add_procs workers")
end


"""
    powers_of_negative_one(power)

Finds -1 ^ power without calling ^

# Examples
```jldoctest
julia> GPLinearODEMaker.powers_of_negative_one(0) == GPLinearODEMaker.powers_of_negative_one(2) == 1
true
julia> GPLinearODEMaker.powers_of_negative_one(1) == GPLinearODEMaker.powers_of_negative_one(3) == -1
true
```
"""
powers_of_negative_one(power::Integer) = iseven(power) ? 1 : -1


"""
    remove_zeros(V)

Return the a version of the passed vector after removing all zero entries.
Could possibly be replaced with a view-based version.

# Examples
```jldoctest
julia> V = [1, 2, 0, 3, 0, 4, 5];

julia> GPLinearODEMaker.remove_zeros(V)
5-element Array{Int64,1}:
 1
 2
 3
 4
 5
```
"""
remove_zeros(V::Vector{T} where T<:Real) = V[findall(!iszero, V)]


"""
    log_laplace_approximation(H, g, logh; λ=1)

Compute the logarithm of the Laplace approxmation for the integral of a function
of the following form
∫ exp(-λ g(y)) h(y) dy ≈ exp(-λ g(y*)) h(y*) (2π/λ)^(d/2) |H(y*)|^(-1/2)
where y* is the value of y at the global mode and H is the (Hessian) matrix of
second order partial derivatives of g(y) (see slide 10 of
http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/laplace.pdf). When used to
calculate evidences, one can set λ = 1, g(y) = -log-likelihood,
h(y) = model prior, and H(y) = the Fisher information matrix (FIM) or the
Hessian matrix of the negative log-likelihood. Could possiblly improve with
methods from Ruli et al. 2016 (https://arxiv.org/pdf/1502.06440.pdf)?
"""
function log_laplace_approximation(
    H::Union{Symmetric,Matrix{T}},
    g::Real,
    logh::Real;
    λ::Real=1
    ) where {T<:Real}

    @assert size(H, 1) == size(H, 2)
    n = size(H, 1)

    return logh - λ * g + 0.5 * (n * log(2 * π / λ) - logdet(H))

end


riffle(list_of_things::Vector{<:Vector{<:Real}}) = collect(Iterators.flatten(zip(list_of_things...)))
unriffle(riffled_list::Vector{<:Real}, n_out::Int) = [riffled_list[i:n_out:end] for i in 1:n_out]


remainder(vec, x) = [i > 0 ? i % x : (i % x) + x for i in vec]


rounded(x::Real) = round(x, digits=2)