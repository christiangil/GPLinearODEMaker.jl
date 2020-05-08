using Test
using LinearAlgebra
import GPLinearODEMaker
GLOM = GPLinearODEMaker

println("Testing...")

@testset "cholesky factorizations" begin
    A = [4. 12 -16; 12 37 -43; -16 -43 98]
    chol_A = GLOM.ridge_chol(A)
    @test chol_A==cholesky(A)==GLOM.symmetric_A(A, chol=true)
    @test chol_A.L==LowerTriangular([2. 0 0;6 1 0;-8 5 3])
    A = [4. 12 -16; 12 37 -43; -16 -43 88]
    @test_logs (:warn, "added a ridge")  GLOM.ridge_chol(A)
    println()
end

@testset "nlogL derivatives" begin
    kernel, n_kern_hyper = include("../src/kernels/se_kernel.jl")

    n = 100
    xs = 20 .* sort(rand(n))
    y1 = sin.(xs) .+ 0.1 .* randn(n)
    noise1 = 0.1 .* ones(n)
    y2 = cos.(xs) .+ 0.2 .* randn(n)
    noise2 = 0.2 .* ones(n)

    ys = collect(Iterators.flatten(zip(y1, y2)))
    noise = collect(Iterators.flatten(zip(noise1, noise2)))

    prob_def = GLOM.GLO(kernel, n_kern_hyper, 2, 2, xs, ys; noise = noise, a0=[[1. 0.1];[0.1 1]])

    @test GLOM.est_dΣdθ(prob_def, 1 .+ rand(n_kern_hyper); return_bool=true, print_stuff=false)
    @test GLOM.test_∇nlogL_GLOM(prob_def, 1 .+ rand(n_kern_hyper), print_stuff=false)
    @test GLOM.test_∇∇nlogL_GLOM(prob_def, 1 .+ rand(n_kern_hyper), print_stuff=false)
    println()
end
