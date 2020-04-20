# using Pkg
# Pkg.activate(".")
import GPLinearODEMaker
GLOM = GPLinearODEMaker
using SymEngine

# process for se_kernel_base
# @vars σ δ λ
# GLOM.kernel_coder(σ * exp(-δ * δ / (2 * λ * λ)), "se")
@vars δ λ
GLOM.kernel_coder(GLOM.se_kernel_base(λ, δ), "se")

@vars δ λ
GLOM.kernel_coder(GLOM.se_kernel_base(λ, δ), "se_white"; add_white_noise=true)

#process for matern52_kernel_base
@vars δ λ
GLOM.kernel_coder(GLOM.matern52_kernel_base(λ, δ), "m52")

@vars δ λ
GLOM.kernel_coder(GLOM.pp_kernel_base(λ, δ), "pp"; cutoff_var="λ")

@vars δ λ
GLOM.kernel_coder(GLOM.pp_kernel_base(λ, δ), "pp_white"; cutoff_var="λ", add_white_noise=true)

@vars δ λ1 λ2 sratio
GLOM.kernel_coder(GLOM.matern52_kernel_base(λ1, δ) + sratio * sratio * GLOM.matern52_kernel_base(λ2, δ), "m52_m52")

@vars δ λ1 λ2 sratio
GLOM.kernel_coder(GLOM.se_kernel_base(λ1, δ) + sratio * sratio * GLOM.se_kernel_base(λ2, δ), "se_se")

# process for periodic_kernel_base
# @vars δ se_P λ
# GLOM.kernel_coder(periodic_kernel_base([se_P, λ], δ), "periodic")
# @vars δ λ
# GLOM.kernel_coder(se_kernel_base(λ, δ), "se"; periodic_var="δ")

# process for quasi_periodic_kernel_base
# @vars δ se_λ qp_P p_λ
# GLOM.kernel_coder(quasi_periodic_kernel_base([se_λ, qp_P, p_λ], δ), "quasi_periodic")
@vars δ δp se_λ p_amp
GLOM.kernel_coder(GLOM.se_kernel_base(se_λ, δ) * GLOM.se_kernel_base(1 / p_amp, δp), "qp"; periodic_var="δp")

#process for rq_kernel_base

# @vars δ k θ
# GLOM.kernel_coder(rq_kernel_base([k, θ], δ), "rq_kernel")
# @vars δ α β
# GLOM.kernel_coder(rq_kernel_base([α, β], δ), "rq_kernel")
@vars δ α μ
GLOM.kernel_coder(GLOM.rq_kernel_base([α, μ], δ), "rq")

@vars δ α μ
GLOM.kernel_coder(GLOM.rm52_kernel_base([α, μ], δ), "rm52")

# @vars δ P λ α
# GLOM.kernel_coder(periodic_rq_kernel_base([P, λ, α], δ), "periodic_rq_kernel")
@vars δ λ α
GLOM.kernel_coder(GLOM.rq_kernel_base([α, λ], δ), "rq"; periodic_var="δ")

@vars δ λ
GLOM.kernel_coder(GLOM.cosine_kernel_base(λ, δ), "cos")
