GPLinearODEMaker.jl
========

GPLinearODEMaker (GLOM) is a package for finding the likelihood (and derivatives thereof) of multivariate Gaussian processes (GP) that are composed of a linear combination of a univariate GP and its derivatives.

![q_0(t) = m_0(t) + a_{00}X(t) + a_{01}\dot{X}(t) + a_{02}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_0(t)%20%3D%20m_0(t)%20%2B%20a_%7B00%7DX(t)%20%2B%20a_%7B01%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B02%7D%5Cddot%7BX%7D(t))

![q_1(t) = m_1(t) + a_{10}X(t) + a_{11}\dot{X}(t) + a_{12}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_1(t)%20%3D%20m_1(t)%20%2B%20a_%7B10%7DX(t)%20%2B%20a_%7B11%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B12%7D%5Cddot%7BX%7D(t))

![\vdots](https://render.githubusercontent.com/render/math?math=%5Cvdots)

![q_l(t) = m_l(t) + a_{l0}X(t) + a_{l1}\dot{X}(t) + a_{l2}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_l(t)%20%3D%20m_l(t)%20%2B%20a_%7Bl0%7DX(t)%20%2B%20a_%7Bl1%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7Bl2%7D%5Cddot%7BX%7D(t))

where each X(t) is the building block GP and the qs are the time series of the outputs.

Here's an example using sine and cosines as the outputs to be modelled. The f, g!, and h! functions at the end give the likelihood, gradient, and Hessian, respectively.

```julia
import GPLinearODEMaker
GLOM = GPLinearODEMaker

kernel, n_kern_hyper = include("../src/kernels/se_kernel.jl")

n = 100
xs = 20 .* sort(rand(n))
noise1 = 0.1 .* ones(n)
noise2 = 0.2 .* ones(n)
y1 = sin.(xs) .+ (noise1 .* randn(n))
y2 = cos.(xs) .+ (noise2 .* randn(n))

ys = collect(Iterators.flatten(zip(y1, y2)))
noise = collect(Iterators.flatten(zip(noise1, noise2)))

prob_def = GLOM.GLO(kernel, n_kern_hyper, 2, 2, xs, ys; noise = noise, a0=[[1. 0.1];[0.1 1]])
total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), [10])
workspace = GLOM.nlogL_matrix_workspace(prob_def, total_hyperparameters)

function f(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, prob_def, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where T<:Real
    G[:] = GLOM.∇nlogL_GLOM!(workspace, prob_def, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
end
function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where T<:Real
    H[:, :] = GLOM.∇∇nlogL_GLOM!(workspace, prob_def, non_zero_hyper)  # feel free to add priors here to optimize on the posterior!
end
```

You can use f, g!, and h! to optimize the GP hyperparameters with external packages like [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or [Flux.jl](https://github.com/FluxML/Flux.jl)

```julia
initial_x = GLOM.remove_zeros(total_hyperparameters)

using Optim

# @time result = optimize(f, initial_x, NelderMead())  # slow or wrong
# @time result = optimize(f, g!, initial_x, LBFGS())  # faster and usually right
@time result = optimize(f, g!, h!, initial_x, NewtonTrustRegion())  # fastest and usually right
```

# Documentation

For more details and options, see the documentation (WIP)

# Installation

The package will be a registered package (by 4/24/2020), and can be installed with `Pkg.add`.

```julia
julia> using Pkg; Pkg.add("GPLinearODEMaker")
```
or through the `pkg` REPL mode by typing
```
] add GPLinearODEMaker
```


# Citation

If you use `GPLinearODEMaker.jl` in your work, please cite the BibTeX entry given in CITATION.bib

The formula images in this README created with [this website](https://tex-image-link-generator.herokuapp.com/)
