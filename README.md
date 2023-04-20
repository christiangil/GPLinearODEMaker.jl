GPLinearODEMaker.jl
========
[![arXiv](https://img.shields.io/badge/arXiv-2009.01085-orange.svg)](https://arxiv.org/abs/2009.01085)
[![DOI](https://zenodo.org/badge/256533350.svg)](https://zenodo.org/badge/latestdoi/256533350)


GPLinearODEMaker (GLOM) is a package for finding the likelihood (and derivatives thereof) of multivariate Gaussian processes (GP) that are composed of a linear combination of a univariate GP and its derivatives.

![q_0(t) = m_0(t) + a_{00}X(t) + a_{01}\dot{X}(t) + a_{02}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_0(t)%20%3D%20m_0(t)%20%2B%20a_%7B00%7DX(t)%20%2B%20a_%7B01%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B02%7D%5Cddot%7BX%7D(t))

![q_1(t) = m_1(t) + a_{10}X(t) + a_{11}\dot{X}(t) + a_{12}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_1(t)%20%3D%20m_1(t)%20%2B%20a_%7B10%7DX(t)%20%2B%20a_%7B11%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B12%7D%5Cddot%7BX%7D(t))

![\vdots](https://render.githubusercontent.com/render/math?math=%5Cvdots)

![q_l(t) = m_l(t) + a_{l0}X(t) + a_{l1}\dot{X}(t) + a_{l2}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_l(t)%20%3D%20m_l(t)%20%2B%20a_%7Bl0%7DX(t)%20%2B%20a_%7Bl1%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7Bl2%7D%5Cddot%7BX%7D(t))

where each X(t) is the latent GP and the qs are the time series of the outputs.

Here's an example using sine and cosines as the outputs to be modelled. The f, g!, and h! functions at the end give the likelihood, gradient, and Hessian, respectively.

```julia
import GPLinearODEMaker; GLOM = GPLinearODEMaker

kernel, n_kern_hyper = GLOM.include_kernel("se")

n = 100
xs = 20 .* sort(rand(n))
noise1 = 0.1 .* ones(n)
noise2 = 0.2 .* ones(n)
y1 = sin.(xs) .+ (noise1 .* randn(n))
y2 = cos.(xs) .+ (noise2 .* randn(n))

ys = collect(Iterators.flatten(zip(y1, y2)))
noise = collect(Iterators.flatten(zip(noise1, noise2)))

glo = GLOM.GLO(kernel, n_kern_hyper, 2, 2, xs, ys; noise = noise, a=[[1. 0.1];[0.1 1]])
total_hyperparameters = append!(collect(Iterators.flatten(glo.a)), [10])
workspace = GLOM.nlogL_matrix_workspace(glo, total_hyperparameters)

function f(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, glo, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where T<:Real
    G[:] = GLOM.∇nlogL_GLOM!(workspace, glo, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
end
function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where T<:Real
    H[:, :] = GLOM.∇∇nlogL_GLOM!(workspace, glo, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
end
```

You can use f, g!, and h! to optimize the GP hyperparameters with external packages like [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or [Flux.jl](https://github.com/FluxML/Flux.jl)

```julia
initial_x = GLOM.remove_zeros(total_hyperparameters)

using Optim

# @time result = optimize(f, initial_x, NelderMead())  # slow or wrong
# @time result = optimize(f, g!, initial_x, LBFGS())  # faster and usually right
@time result = optimize(f, g!, h!, initial_x, NewtonTrustRegion())  # fastest and usually right

fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(glo, result.minimizer)
```

Once you have the best fit hyperparameters, you can easily calculate the GP conditioned on the data (i.e. the GP posterior)

```julia
n_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(glo.x_obs))))
x_samp = collect(range(minimum(glo.x_obs); stop=maximum(glo.x_obs), length=n_samp_points))
n_total_samp_points = n_samp_points * glo.n_out
n_meas = length(glo.x_obs)

mean_GP, σ, mean_GP_obs, Σ = GLOM.GP_posteriors(glo, x_samp, fit_total_hyperparameters; return_mean_obs=true)
```

and use Plots to visualize the results

```julia
using Plots

function make_plot(output::Integer, label::String)
    sample_output_indices = output:glo.n_out:n_total_samp_points
    obs_output_indices = output:glo.n_out:length(ys)
    p = scatter(xs, ys[obs_output_indices], yerror=noise1, label=label)
    plot!(x_samp, mean_GP[sample_output_indices]; ribbon=σ[sample_output_indices], alpha=0.3, label="GP")
    return p
end

plot(make_plot(1, "Sin"), make_plot(2, "Cos"), layout=(2,1), size=(960,540))

```

# Documentation

For more details and options, see the [documentation](https://christiangil.github.io/GPLinearODEMaker.jl/dev)

You can read about the first usage of this package in [our paper](https://arxiv.org/abs/2009.01085)

Also check out our [companion repository](https://github.com/christiangil/GLOM_RV_Example) which has some examples of using GLOM to fit stellar variability and planets

# Installation

The most current, tagged version of [GPLinearODEMaker.jl](https://github.com/christiangil/GPLinearODEMaker.jl) can be easily installed using Julia's Pkg

```julia
Pkg.add("GPLinearODEMaker")
```

If you would like to contribute to the package, or just want to run the latest (untagged) version, you can use the following

```julia
Pkg.develop("GPLinearODEMaker")
```

# Citation

If you use `GPLinearODEMaker.jl` in your work, please cite the BibTeX entry given in CITATION.bib

The formula images in this README created with [this website](https://tex-image-link-generator.herokuapp.com/)
