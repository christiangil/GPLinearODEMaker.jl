"Radial basis function GP kernel (aka squared exonential, ~gaussian)"
function se_kernel_base(λ::Number, δ::Number)
    return exp(-δ * δ / (2 * λ * λ))
end


# """
# Periodic kernel (for random cyclic functions)
# SE kernel where δ^2 -> 4 sin(π δ/P)^2
# """
# function periodic_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
#     P, λ = hyperparameters
#
#     sin_τ = sin(π * δ / P)
#     return exp(-2 * sin_τ * sin_τ / (λ * λ))
# end


# "Quasi-periodic kernel"
# function quasi_periodic_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 3 "incompatible amount of hyperparameters passed"
#     SE_λ, P_P, P_λ = hyperparameters
#
#     return se_kernel_base(SE_λ, δ) * periodic_kernel_base([P_P, P_λ], δ)
# end


# "Ornstein–Uhlenbeck (Exponential) kernel"
# function exp_kernel_base(λ::Number, δ::Number)
#     return exp(-abs(δ) / λ)
# end


# "Exponential-periodic kernel"
# function exp_periodic_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 3 "incompatible amount of hyperparameters passed"
#     OU_λ, P_P, P_λ = hyperparameters
#
#     return ou_kernel_base(OU_λ, δ) * periodic_kernel_base([P_P, P_λ], δ)
# end


# "general Matern kernel"
# function matern_kernel_base(λ::Number, δ::Number, nu::Number)
#
#     #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
#     if δ == 0
#         return kernel_amplitude * kernel_amplitude
#     else
#         x = (sqrt(2 * nu) * abs(δ)) / λ
#         return ((2 ^ (1 - nu)) / (gamma(nu))) * x ^ nu * besselk(nu, x)
#     end
# end


# "Matern 3/2 kernel"
# function matern32_kernel_base(λ::Number, δ::Number)
#     x = sqrt(3) * abs(δ) / λ
#     return (1 + x) * exp(-x)
# end


"Matern 5/2 kernel"
function matern52_kernel_base(λ::Number, δ::Number)
    x = sqrt(5) * abs(δ) / λ
    return (1 + x * (1 + x / 3)) * exp(-x)
end


# "Matern 7/2 kernel"
# function matern72_kernel_base(λ::Number, δ::Number)
#     x = sqrt(7) * abs(δ) / λ
#     return (1 + x * (1 + x * (2 / 5 + x / 15))) * exp(-x)
# end


# "Matern 9/2 kernel"
# function matern92_kernel_base(λ::Number, δ::Number)
#     x = 3 * abs(δ) / λ
#     return (1 + x * (1 + x * (3 / 7 + x * (2 / 21 + x / 105)))) * exp(-x)
# end

"peicewise polynomial kernel that is twice MS differentiable. See eq 4.21 in RW"
function pp_kernel_base(λ::Number, δ::Number)
    # D = 1  # 1 dimension
    # q = 2  # twice MS differentiable
    # j = q + 1 + floor(D / 2) = 3
    r = abs(δ) / λ
    return (1 - r) ^ 5 * (8 * r * r + 5 * r + 1)
end


# """
# Rational Quadratic kernel
# Equivalent to adding together SE kernels with the inverse square of the
# lengthscales (τ = SE_λ^-2) are distributed as a Gamma distribution of p(τ|k,θ)
# where k (sometimes written as α) is the shape parameter and θ is the scale
# parameter. When α→∞, the RQ is identical to the SE with λ = (k θ)^-1/2.
# """
# function rq_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
#     k, θ = hyperparameters
#
#     return (1 + δ * δ * θ / 2) ^ -k
#     # return (1 / θ + δ * δ / 2) ^ -k
# end
# """
# Rational Quadratic kernel
# Equivalent to adding together SE kernels with the inverse square of the
# lengthscales (τ = SE_λ^-2) are distributed as a Gamma distribution of p(τ|α,β)
# where α (sometimes written as k) is the shape parameter and β is the rate
# parameter. When α→∞, the RQ is identical to the SE with λ = (α / β)^-1/2.
# """
# function rq_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
#     α, β = hyperparameters
#
#     # return (1 + δ * δ / (2 * β)) ^ -α
#     return (β + δ * δ / 2) ^ -α
# end
"""
Rational Quadratic kernel
Equivalent to adding together SE kernels with the inverse square of the
lengthscales (τ = SE_λ^-2) are distributed as a Gamma distribution of p(τ|α,μ)
where α (sometimes written as k) is the shape parameter and μ is the mean of the
distribution. When α→∞, the RQ is identical to the SE with λ = μ^-1/2.
"""
function rq_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)

    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    α, μ = hyperparameters

    return (1 + δ * δ * μ / (2 * α)) ^ -α
    # return (α / μ + δ * δ / 2) ^ -α
end


"""
Rational Matern 5/2 kernel
Equivalent to adding together Matern 5/2 kernels with the inverse of the
lengthscale (τ = M52_λ^-1) are distributed as a Gamma distribution of p(τ|α,μ)
where α (sometimes written as k) is the shape parameter and μ is the mean of the
distribution.
"""
function rm52_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    α, μ = hyperparameters

    x = sqrt(5) * abs(δ)
    y = x + α / μ
    return y ^ -α * (α * (α + x * (2 + α) * μ) + x * x * (1 + α) * (3 + α) * μ * μ / 3) / (y * y * μ * μ)
end


# """
# Periodic kernel (for random cyclic functions)
# RQ kernel where δ^2 -> 4 sin(π δ/P)^2
# """
# function periodic_rq_kernel_base(hyperparameters::Vector{<:Number}, δ::Number)
#
#     @assert length(hyperparameters) == 3 "incompatible amount of hyperparameters passed"
#     P, λ, α = hyperparameters
#
#     sin_τ = sin(π * δ / P)
#     return (1 + 2 * sin_τ * sin_τ / (α * λ * λ)) ^ -α
# end


# """
# Bessel (function of the first kind) kernel
# Bessel functions of the first kind, denoted as Jα(x), are solutions of Bessel's
# differential equation that are finite at the origin (x = 0) for integer or positive α
# http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
# """
# function bessel_kernel_base(hyperparameters::Vector{<:Number}, δ::Number; nu=0)
#
#     @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
#     λ, n = hyperparameters
#
#     @assert nu >= 0 "nu must be >= 0"
#
#     return besselj(nu + 1, λ * δ) / (δ ^ (-n * (nu + 1)))
# end
