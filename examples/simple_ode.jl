using Pkg
Pkg.activate(".")
import GPLinearODEMaker
GLOM = GPLinearODEMaker

kernel, n_kern_hyper = include("../src/kernels/pp_kernel.jl")

n = 200
xs = 20 .* sort(rand(n))
y1 = sin.(xs) .+ 0.1 .* randn(n)
noise1 = 0.1 .* ones(n)
y2 = cos.(xs) .+ 0.2 .* randn(n)
noise2 = 0.2 .* ones(n)

ys = collect(Iterators.flatten(zip(y1, y2)))
noise = collect(Iterators.flatten(zip(noise1, noise2)))

prob_def = GLOM.GLO(kernel, n_kern_hyper, 2, 2, xs, ys; noise = noise, a0=[[1. 0.1];[0.1 1]])
total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), [10])
total_hyperparameters .*= (1 .+ 0.1 .* rand(length(total_hyperparameters)))
workspace = GLOM.nlogL_matrix_workspace(prob_def, total_hyperparameters)

using Optim

initial_x = GLOM.remove_zeros(total_hyperparameters)
f(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, prob_def, non_zero_hyper)
function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where T<:Real
    G[:] = GLOM.∇nlogL_GLOM!(workspace, prob_def, non_zero_hyper)
end
function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where T<:Real
    H[:, :] = GLOM.∇∇nlogL_GLOM!(workspace, prob_def, non_zero_hyper)
end

# @time result = optimize(f, initial_x, NelderMead()) # 26s
# @time result = optimize(f, g!, initial_x, LBFGS()) # 40s
@time result = optimize(f, g!, h!, initial_x, NewtonTrustRegion()) # 15s

fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(prob_def, result.minimizer)

n_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(prob_def.x_obs))))
x_samp = collect(range(minimum(prob_def.x_obs); stop=maximum(prob_def.x_obs), length=n_samp_points))
n_total_samp_points = n_samp_points * prob_def.n_out
n_meas = length(prob_def.x_obs)

mean_GP, σ, mean_GP_obs, Σ = GLOM.GP_posteriors(prob_def, x_samp, total_hyperparameters; return_mean_obs=true)

n_show = 5
show_curves = zeros(n_show, n_total_samp_points)
L = GLOM.ridge_chol(Σ).L
for i in 1:n_show
    show_curves[i, :] = L * randn(n_total_samp_points) + mean_GP
end

using Plots

function make_plot(output::Integer)
    sample_output_indices = output:prob_def.n_out:n_total_samp_points
    obs_output_indices = output:prob_def.n_out:length(ys)
    p = scatter(xs, ys[obs_output_indices], yerror=noise1, leg=false)
    for i in 1:n_show
        plot!(x_samp, show_curves[i, sample_output_indices], leg=false)
    end
    plot!(x_samp, mean_GP[sample_output_indices]; ribbon=σ[sample_output_indices], alpha = 0.3, leg=false)
    return p
end

plot(make_plot(1), make_plot(2), layout=(2,1))

savefig("test.png")


## Assuming the noise is inherent to the process instead of the measurement

kernel, n_kern_hyper = include("../src/kernels/pp_white_kernel.jl")

prob_def = GLOM.GLO(kernel, n_kern_hyper, 2, 2, xs, ys; a0=[[1. 0.1];[0.1 1]])
total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), [10, 0.1])
total_hyperparameters .*= (1 .+ 0.1 .* rand(length(total_hyperparameters)))
workspace = GLOM.nlogL_matrix_workspace(prob_def, total_hyperparameters)

initial_x = GLOM.remove_zeros(total_hyperparameters)

# @time result = optimize(f, initial_x, NelderMead()) # 26s
# @time result = optimize(f, g!, initial_x, LBFGS()) # 40s
@time result = optimize(f, g!, h!, initial_x, NewtonTrustRegion()) # 15s

fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(prob_def, result.minimizer)

mean_GP, σ, mean_GP_obs, Σ = GLOM.GP_posteriors(prob_def, x_samp, total_hyperparameters; return_mean_obs=true)

L = GLOM.ridge_chol(Σ).L
for i in 1:n_show
    show_curves[i, :] = L * randn(n_total_samp_points) + mean_GP
end

plot(make_plot(1), make_plot(2), layout=(2,1))

savefig("test.png")
