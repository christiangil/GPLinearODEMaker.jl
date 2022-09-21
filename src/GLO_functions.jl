import Base.copy

"""
	GLO{T1<:Real, T2<:Integer} a.k.a. GPLinearODE

A structure that holds all of the relevant information for constructing the
GLOM model

# Arguments
- `kernel::Function`: kernel function for the latent Gaussian process (GP)
- `n_kern_hyper::T2`: amount of hyperparameters for the kernel function
- `n_dif::T2`: amount of times you are differenting the base kernel
- `n_out::T2`: amount of outputs you are jointly modelling
- `x_obs::Vector{T1}`: the observation times/phases
- `y_obs::Vector{T1}`: the flattened, observed data
- `noise::Vector{T1}`: the measurement noise at all observations (σ, not σ^2)
- `normals::Vector{T1}`: the normalization of each section of y_obs
- `a::Matrix{T1}`: the meta kernel coefficients
- `non_zero_hyper_inds::Vector{T2}`: the indices of the non-zero hyperparameters
- `coeff_orders::AbstractArray{T2,6}`: The powers that each `a` coefficient is taken to for each part of the covariance matrix construction. Used for constructing differentiated versions of the kernel
- `coeff_coeffs::AbstractArray{T2,4}`: The coefficients that each `a` coefficient is multiplied by for each part of the covariance matrix construction. Used for constructing differentiated versions of the kernel
- `covariance::Array{T1, 3}`: the measurement covariance at all observations
- `has_covariance::Bool`: Whether or not there are measurement covariances (as opposed to just having measurement noises)
- `kernel_changes_with_output::Bool`: A flag for if the kernel is not the same for each output (e.g. there is a time delay kernel hyperparameter that changes)
"""
struct GLO{T1<:Real, T2<:Integer}
	kernel::Function
	n_kern_hyper::T2
	n_dif::T2
	n_out::T2
	x_obs::Vector{T1}
	y_obs::Vector{T1}
	noise::Vector{T1}
	normals::Vector{T1}
	a::Matrix{T1}
	non_zero_hyper_inds::Vector{T2}
	coeff_orders::AbstractArray{T2,6}
	coeff_coeffs::AbstractArray{T2,4}
	covariance::Array{T1, 3}
	has_covariance::Bool
	kernel_changes_with_output::Bool

	function GLO(
		kernel::Function,
		n_kern_hyper::T2,
		n_dif::T2,
		n_out::T2,
		x_obs::Vector{T1},
		y_obs::Vector{T1};
		noise::Vector{T1} = zeros(length(y_obs)),
		normals::Vector{T1} = ones(n_out),
		a::Matrix{T1} = ones(n_out, n_dif),
		covariance::Array{T1, 3} = zeros(length(x_obs), n_out, n_out),
		kernel_changes_with_output::Bool=false
		) where {T1<:Real, T2<:Integer}

		@assert size(a) == (n_out, n_dif)
		non_zero_hyper_inds = append!(findall(!iszero, collect(Iterators.flatten(a))), collect(1:n_kern_hyper) .+ length(a))
		coeff_orders, coeff_coeffs = coefficient_orders(n_out, n_dif, a=a)
		has_covariance = (covariance != zeros(length(x_obs), n_out, n_out))

		return GLO(kernel, n_kern_hyper, n_dif, n_out, x_obs, y_obs, noise, normals, a, non_zero_hyper_inds, coeff_orders, coeff_coeffs, covariance, has_covariance, kernel_changes_with_output)
	end
	function GLO(
		kernel::Function,  # kernel function
		n_kern_hyper::T2,  # amount of hyperparameters for the kernel function
		n_dif::T2,  # amount of times you are differenting the base kernel
		n_out::T2,  # amount of scores you are jointly modelling
		x_obs::Vector{T1}, # the observation times/phases
		y_obs::Vector{T1},  # the flattened, observed data
		noise::Vector{T1},  # the measurement noise at all observations
		normals::Vector{T1},  # the normalization of each section of y_obs
		a::Matrix{T1},
		non_zero_hyper_inds::Vector{T2},
		coeff_orders::Array{T2,6},
		coeff_coeffs::Array{T2,4},
		covariance::Array{T1, 3},
		has_covariance::Bool,
		kernel_changes_with_output::Bool) where {T1<:Real, T2<:Integer}

		@assert isfinite(kernel(ones(n_kern_hyper), randn(), zeros(Int64, 2 + n_kern_hyper)))  # make sure the kernel is valid by testing a sample input
		@assert 0 < n_dif <= 3
		@assert 0 < n_out
		n_meas = length(x_obs)
		@assert (n_meas * n_out) == length(y_obs) == length(noise)
		@assert n_meas == size(covariance, 1)
		@assert length(normals) == n_out
		@assert size(a) == (n_out, n_dif)
		@assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)  # maybe unnecessary due to the fact that we construct it
		@assert size(coeff_coeffs) == (n_out, n_out, n_dif, n_dif)  # maybe unnecessary due to the fact that we construct it
		@assert length(non_zero_hyper_inds) == length(findall(!iszero, collect(Iterators.flatten(a)))) + n_kern_hyper
		@assert n_out == size(covariance, 2) == size(covariance, 3)
		@assert (covariance != zeros(n_meas, n_out, n_out)) == has_covariance

		return new{typeof(x_obs[1]),typeof(n_kern_hyper)}(kernel, n_kern_hyper, n_dif, n_out, x_obs, y_obs, noise, normals, a, non_zero_hyper_inds, coeff_orders, coeff_coeffs, covariance, has_covariance, kernel_changes_with_output)
	end
end
Base.copy(glo::GLO) = GLO(glo.kernel, glo.n_kern_hyper, glo.n_dif, glo.n_out, copy(glo.x_obs), copy(glo.y_obs), copy(glo.noise), copy(glo.normals), copy(glo.a), copy(glo.non_zero_hyper_inds), copy(glo.coeff_orders), copy(glo.coeff_coeffs), copy(glo.covariance), glo.has_covariance, glo.kernel_changes_with_output)

"""
	normalize_GLO!(glo)

Standardize the outputs in `glo.y_obs` (and `glo.noise`) to have 0-mean and unit
variance and store the normalization in `glo.normals`
"""
function normalize_GLO!(glo::GLO)
	renorms = ones(glo.n_out)
	for i in 1:glo.n_out
		inds = (i:glo.n_out:length(glo.y_obs))
		glo.y_obs[inds] .-= mean(glo.y_obs[inds])
		renorms[i] = std(glo.y_obs[inds])
	end
	normalize_GLO!(glo, renorms)
end

"""
	normalize_GLO!(glo, renorms)

Divide the outputs in `glo.y_obs` and `glo.noise` and multiply `glo.normals` by
renorms
"""
function normalize_GLO!(glo::GLO, renorms::Vector)
	@assert length(renorms) == glo.n_out
	for i in 1:glo.n_out
		inds = (i:glo.n_out:length(glo.y_obs))
		glo.normals[i] *= renorms[i]
		glo.y_obs[inds] /= renorms[i]
		glo.noise[inds] /= renorms[i]
	end
	if glo.has_covariance
		renorm_mat = renorms .* transpose(renorms)
		for i in 1:length(glo.x_obs)
			glo.covariance[i, :, :] ./= renorm_mat
		end
	end
end
