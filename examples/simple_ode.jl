using Pkg
Pkg.activate("examples")
Pkg.instantiate()

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

using Optim

initial_x = GLOM.remove_zeros(total_hyperparameters)
f(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, glo, non_zero_hyper)
function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where T<:Real
    G[:] = GLOM.∇nlogL_GLOM!(workspace, glo, non_zero_hyper)
end
function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where T<:Real
    H[:, :] = GLOM.∇∇nlogL_GLOM!(workspace, glo, non_zero_hyper)
end

function optim_cb(x::OptimizationState)
    println()
    if x.iteration > 0
        println("Iteration:     ", x.iteration)
        println("Time so far:   ", x.metadata["time"], " s")
        println("Currnet score: ", x.value)
        println("Gradient norm: ", x.g_norm)
        println()
    end
    return false
end

# @time result = optimize(f, initial_x, NelderMead()) # 26s
# @time result = optimize(f, g!, initial_x, LBFGS()) # 40s
@time result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=50)) # 15s

fit_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(glo, result.minimizer)

n_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(glo.x_obs))))
x_samp = collect(range(minimum(glo.x_obs); stop=maximum(glo.x_obs), length=n_samp_points))
n_total_samp_points = n_samp_points * glo.n_out
n_meas = length(glo.x_obs)

mean_GP, σ, mean_GP_obs, Σ = GLOM.GP_posteriors(glo, x_samp, fit_total_hyperparameters; return_mean_obs=true)

#=
n_show = 5
show_curves = zeros(n_show, n_total_samp_points)
L = GLOM.ridge_chol(Σ).L
for i in 1:n_show
    show_curves[i, :] = L * randn(n_total_samp_points) + mean_GP
end

using Plots

function make_plot(output::Integer, label::String; show_draws::Bool=false)
    sample_output_indices = output:glo.n_out:n_total_samp_points
    obs_output_indices = output:glo.n_out:length(ys)
    p = scatter(xs, ys[obs_output_indices], yerror=noise1, label=label)
    if show_draws
        for i in 1:n_show
            plot!(x_samp, show_curves[i, sample_output_indices], leg=false)
        end
    end
    plot!(x_samp, mean_GP[sample_output_indices]; ribbon=σ[sample_output_indices], alpha=0.3, label="GP")
    return p
end

plot(make_plot(1, "Sin"), make_plot(2, "Cos"), layout=(2,1), size=(960,540))
savefig("example/simple_ode.png")
=#
