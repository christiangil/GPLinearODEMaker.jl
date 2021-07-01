"""
    log_inverse_gamma(x; α=1., β=1., d=0)

Log of the InverseGamma PDF.
Equivalent to `using Distributions; logpdf(InverseGamma(α, β), x)`
https://en.wikipedia.org/wiki/Inverse-gamma_distribution

# Keyword Arguments
- `x::Real`: Input
- `α::Real=1.`: Shape parameter
- `β::Real=1.`: Scale parameter
- `d::Integer=0`: How many derivatives to take
"""
function log_inverse_gamma(x::Real; α::Real=1., β::Real=1., d::Integer=0)
    @assert 0 <= d <= 2 "Can only differentiate up to two times"
    @assert α > 0 "α needs to be > 0"
    @assert β > 0 "β needs to be > 0"
    if d == 0
        x > 0 ? val = -(β / x) - (1 + α) * log(x) + α * log(β) - loggamma(α) : val = -Inf
    elseif d == 1
        x > 0 ? val = (β / x - (1 + α)) / x : val = 0
    else
        x > 0 ? val = (-2 * β / x + (1 + α)) / (x * x) : val = 0
    end
    return val
end


"""
    log_gamma(x, parameters; d=0, passed_mode_std=false)

Log of the Gamma PDF.
Equivalent to `using Distributions; logpdf(Gamma(α, θ), x)`
https://en.wikipedia.org/wiki/Gamma_distribution

# Arguments
- `x::Real`: Input
- `parameters::Vector`: Either the shape and scale parameters (i.e. [α, θ]) or a mode and standard deviation
- `d::Integer=0`: How many derivatives to take
- `passed_mode_std::Bool=false`: Whether to the `parameters` need to be converted to shape and scale parameters
"""
function log_gamma(x::Real, parameters::Vector{<:Real}; d::Integer=0, passed_mode_std::Bool=false)
    @assert 0 <= d <= 2 "Can only differentiate up to two times"
    @assert length(parameters) == 2
    assert_positive(parameters)
    if passed_mode_std
        parameters = gamma_mode_std_to_α_θ(parameters[1], parameters[2])
    end
    α = parameters[1]
    θ = parameters[2]
    if d == 0
        x > 0 ? val = -(x / θ) + (α - 1) * log(x) - α * log(θ) - loggamma(α) : val = -Inf
    elseif d == 1
        x > 0 ? val = (α - 1) / x - 1 / θ : val = 0
    else
        x > 0 ? val = -(α - 1) / (x * x) : val = 0
    end
    return val
end


"""
    gamma_mode_std_to_α_θ(m, σ)

Convert a gamma distribution's mode and standard deviation into it's shape and
scale parameters

# Arguments
- `m::Real`: Desired mode of gamma distribution
- `σ::Real`: Desired standard deviation of gamma distribution
"""
function gamma_mode_std_to_α_θ(m::Real, σ::Real)
    θ = (sqrt(m ^ 2 + 4 * σ ^ 2) - m) / 2
    α = m / θ + 1
    return [α, θ]
end


"""
    gauss_cdf(x)

The CDF of a Gaussian (i.e. (1 + erf(x))/2)
"""
gauss_cdf(x::Real) = (1 + erf(x))/2


"""
    log_gaussian(x, parameters; d=0, min=-Inf, max=Inf)

Log of the Gaussian PDF.
Equivalent to `using Distributions; logpdf(Gaussian(μ, σ), x)`

# Arguments
- `x::Real`: Input
- `parameters::Vector`: The mean and standard deviation parameters (i.e. [μ, σ])
- `d::Integer=0`: How many derivatives to take
- `min::Real=-Inf`: Where to minimally truncate the Gaussian
- `max::Real=Inf`: Where to maximally truncate the Gaussian
"""
function log_gaussian(x::Real, parameters::Vector{<:Real}; d::Integer=0, min::Real=-Inf, max::Real=Inf)
    @assert 0 <= d <= 2 "Can only differentiate up to two times"
    @assert length(parameters) == 2
    μ = parameters[1]
    σ = parameters[2]
    assert_positive(σ)
    @assert min < max
    if d == 0
        (min < x < max) ? val = (-((x - μ) / σ) ^ 2 - log(2 * π * σ * σ)) / 2 - log(1 - gauss_cdf(min - μ) - gauss_cdf(μ - max)) : val = -Inf
    elseif d == 1
        (min < x < max) ? val = -(x - μ)/(σ * σ) : val = 0
    else
        (min < x < max) ? val = -1 / (σ * σ) : val = 0
    end
    return val
end


"""
    log_uniform(x; d=0, min=0, max=1)

Log of the Uniform PDF.

# Arguments
- `x::Real`: Input
- `min_max::Vector=[0,1]`: Where to truncate the Uniform
- `d::Integer=0`: How many derivatives to take
"""
function log_uniform(x::Real; min_max::Vector{<:Real}=[0,1], d::Integer=0)
    @assert 0 <= d <= 2 "Can only differentiate up to two times"
    @assert length(min_max) == 2
    min, max = min_max
    @assert min < max
    if d == 0
        min <= x <= max ? -log(max - min) : -Inf
    else
        return 0
    end
end


"""
    log_loguniform(x::Real, min_max::Vector{<:Real}; d::Integer=0, shift::Real=0)

Log of the log-Uniform PDF. Flattens out in log space starting at shift
Also known as a (modified in shifted case) Jeffrey's prior

# Arguments
- `x::Real`: Input
- `min_max::Vector`: Where to truncate the log-Uniform
- `d::Integer=0`: How many derivatives to take
- `shift::Real=0`: Where to shift the peak of the distribution
"""
function log_loguniform(x::Real, min_max::Vector{<:Real}; d::Integer=0, shift::Real=0)
    @assert 0 <= d <= 2 "Can only differentiate up to two times"
    @assert length(min_max) == 2
    min, max = min_max
    @assert 0 < min + shift < max + shift
    xpshift = x + shift
    if d == 0
        min <= x <= max ? val = -log(xpshift) - log(log((max + shift)/(min + shift))) : val = -Inf
    elseif d == 1
        min <= x <= max ? val = -1 / xpshift : val = 0
    elseif d == 2
        min <= x <= max ? val = 1 / (xpshift * xpshift) : val = 0
    end
    return val
end


"""
    log_Rayleigh(x, σ; d=0, cutoff=Inf)

Log of the Rayleigh PDF.
https://en.wikipedia.org/wiki/Rayleigh_distribution

# Arguments
- `x::Real`: Input
- `σ::Real`: The mode
- `d::Integer=0`: How many derivatives to take
- `cutoff::Real=Inf`: Where to cutoff the tail of the distribution
"""
function log_Rayleigh(x::Real, σ::Real; d::Integer=0, cutoff::Real=Inf)
    @assert 0 <= d <= 2
    @assert cutoff > 0
    cutoff == Inf ? normalization = 0 : normalization = -log(1 - exp(-cutoff * cutoff / 2 / σ / σ))
    if d == 0
        0 <= x <= cutoff ? val = normalization - (x * x / 2 / σ / σ) + log(x / σ / σ) : val = -Inf
    elseif d == 1
        0 <= x <= cutoff ? val = 1 / x - x / σ / σ : val = 0
    elseif d == 2
        0 <= x <= cutoff ? val = -1 / x / x - 1 / σ / σ : val = 0
    end
    return val
end


"""
    log_circle(xs, min_max_r; d=[0,0])

Log of the 2D circle PDF.

# Arguments
- `xs::Real`: Inputs
- `min_max_r::Vector{<:Real}`: How far the inner and outer edge of the circle extends from the origin
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
"""
function log_circle(xs::Vector{<:Real}, min_max_r::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) == 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(xs) == length(min_max_r) == length(d) == 2
    min_r, max_r = min_max_r

    # @assert min_r < sqrt(dot(x, x)) < max_r
    # @assert min_θ < atan(x[1], x[2]) < max_θ
    if all(d .== 0)
        min_r < sqrt(dot(xs, xs)) < max_r ? val = -log(2 * π * (max_r ^ 2 - min_r ^ 2)) : val = -Inf
    else
        return 0
    end
    return val
end


"""
    log_cone(xs; d=[0,0])

Log of the 2D unit cone PDF.

# Arguments
- `xs::Real`: Inputs
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
"""
function log_cone(xs::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(xs) == length(d) == 2

    r_sq = dot(xs, xs)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(3 / π * (1 - r)) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = xs[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = (-xs[2] ^ 2 * r + xs[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = xs[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (xs[1] * xs[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = (-xs[1] ^ 2 * r + xs[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
    log_quad_cone(xs; d=[0,0])

Log of the 2D unit quadratic cone PDF.

# Arguments
- `xs::Real`: Inputs
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
"""
function log_quad_cone(xs::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(xs) == length(d) == 2

    r_sq = dot(xs, xs)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(6 / π * (1 - 2 * r + r_sq)) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = 2 * xs[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = 2 * (-xs[2] ^ 2 * r + xs[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = 2 * xs[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (2 * xs[1] * xs[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = 2 * (-xs[1] ^ 2 * r + xs[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
    log_cubic_cone(xs; d=[0,0])

Log of the 2D unit cubic cone PDF.

# Arguments
- `xs::Real`: Inputs
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
"""
function log_cubic_cone(xs::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(xs) == length(d) == 2

    r_sq = dot(xs, xs)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(10 / π * (1 - r)^3) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = 3 * xs[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = 3 * (-xs[2] ^ 2 * r + xs[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = 3 * xs[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (3 * xs[1] * xs[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = 3 * (-xs[1] ^ 2 * r + xs[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
    log_rot_Rayleigh(xs; d=[0,0], σ=1/5, cutoff=Inf)

Log of the 2D rotated Rayleigh PDF

# Arguments
- `xs::Real`: Inputs
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
- `σ::Real=1/5`: the radial mode of the distribution
- `cutoff::Real=Inf`: Where to cutoff the tail of the distribution
"""
function log_rot_Rayleigh(xs::Vector{<:Real}, σ::Real; d::Vector{<:Integer}=[0,0], cutoff::Real=Inf)
    @assert all(0 .<= d .<= 2)
    @assert sum(d) <= 2
    @assert cutoff > 0

    @assert length(xs) == length(d) == 2
    r_sq = dot(xs, xs)  # x^2 + y^2
    r = sqrt(r_sq)
    σ_sq = σ ^ 2
    cutoff == Inf ? normalization = -log(2 * σ * σ * π * π * π) / 2 : normalization = -log(2 * π) - log(sqrt(π / 2) * σ * erf(cutoff / sqrt(2) / σ) - cutoff * exp(- cutoff * cutoff / 2 / σ / σ))
    if d == [0,0]
        0 <= r < cutoff ? val = normalization - r_sq / (2 * σ_sq) + log(r) - (2 * log(σ)) : val = -Inf
    elseif d == [0,1]
        0 <= r < cutoff ? val = -xs[2] * (r_sq - σ_sq) / (r_sq * σ_sq) : val = 0
    elseif d == [0,2]
        0 <= r < cutoff ? val = -(xs[1] ^ 4 + xs[1] ^ 2 * (2 * xs[2] ^ 2 - σ_sq) + xs[2] ^ 2 * (xs[2] ^ 2 + σ_sq)) / (r_sq ^ 2 * σ_sq) : val = 0
    elseif d == [1,0]
        0 <= r < cutoff ? val = -xs[1] * (r_sq - σ_sq) / (r_sq * σ_sq) : val = 0
    elseif d == [1,1]
        0 <= r < cutoff ? val = -2 * xs[1] * xs[2] / r_sq ^ 2 : val = 0
    elseif d == [2,0]
        0 <= r < cutoff ? val = -(xs[2] ^ 4 + xs[2] ^ 2 * (2 * xs[1] ^ 2 - σ_sq) + xs[1] ^ 2 * (xs[1] ^ 2 + σ_sq)) / (r_sq ^ 2 * σ_sq) : val = 0
    end
    return val
end


"""
    log_bvnormal(xs, Σ; μ=zeros(T, 2), d=[0,0], lows=zeros(T, 2) .- Inf)

Log of the bivariate normal PDF
NOTE THAT THAT WHEN USING lows!=[-∞,...], THIS IS NOT PROPERLY NORMALIZED

# Arguments
- `xs::Real`: Inputs
- `Σ::Colesky`: The covariance matrix of the distribution
- `μ::Vector=zeros(T, 2)`: The mean of the distribution
- `d::Vector{<:Integer}=[0,0]`: How many derivatives to take
- `lows::Vector=zeros(T, 2) .- Inf`: The lower cutoffs of the distribution
"""
function log_bvnormal(xs::Vector{T}, Σ::Cholesky; μ::Vector{T}=zeros(T, 2), d::Vector{<:Integer}=[0,0], lows::Vector{T}=zeros(T, 2) .- Inf) where {T<:Real}
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(xs) == length(d) == length(μ) == 2
    y = xs - μ

    if sum(d) == 0
        all(xs .>= lows) ? val = -nlogL(Σ, y) : val = -Inf
    elseif sum(d) == 1
        y1 = Float64.(d)
        all(xs .>= lows) ? val = -dnlogLdθ(y1, Σ \ y) : val = 0
    elseif d == [0,2]
        y1 = y2 = [0, 1.]
        all(xs .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    elseif d == [1,1]
        y1 = [1, 0.]
        y2 = [0, 1.]
        all(xs .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    elseif d == [2,0]
        y1 = y2 = [1, 0.]
        all(xs .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    end
    return val
end


"""
    bvnormal_covariance(σ11, σ22, ρ)

Converting 2 standard deviations and their correlation into a bivariate
covariance matrix and Cholesky factorizing the result.
"""
function bvnormal_covariance(σ11::Real, σ22::Real, ρ::Real)
    assert_positive(σ11, σ22)
    @assert 0 <= ρ <= 1
    v12 = σ11*σ22*ρ
    return ridge_chol([σ11^2 v12;v12 σ22^2])
end
