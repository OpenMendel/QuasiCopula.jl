module GaussianVCBenchmark

using InteractiveUtils, LinearAlgebra, Profile, Random
using BenchmarkTools, GLMCopula, GLM

Random.seed!(123)

n = 100  # number of observations
ns = rand(100:300, n) # ni in each observation
p = 3   # number of mean parameters
m = 2   # number of variance components
d = Normal()
D = typeof(d)
gcs = Vector{GaussianCopulaVCObs{Float64, D}}(undef, n)
# true parameter values
βtruth = ones(p)
σ2truth = collect(1.:m)
σ02truth = 1.0
for i in 1:n
    ni = ns[i]
    # set up covariance matrix
    V1 = convert(Matrix, Symmetric([Float64(i * (ni - j + 1)) for i in 1:ni, j in 1:ni])) # a pd matrix
    V1 ./= norm(V1) / sqrt(ni) # scale to have Frobenius norm sqrt(n)
    prob = fill(1/ni, ni)
    V2 = ni .* (Diagonal(prob) - prob * transpose(prob))
    V2 ./= norm(V2) / sqrt(ni) # scale to have Frobenious norm sqrt(n)
    Ω = σ2truth[1] * V1 + σ2truth[2] * V2 + σ02truth * I
    Ωchol = cholesky(Symmetric(Ω))
    # simulate design matrix
    X = [ones(ni) randn(ni, p-1)]
    # generate responses
    y = X * βtruth + Ωchol.L * randn(ni)
    # add to data
    gcs[i] = GaussianCopulaVCObs(y, X, [V1, V2], d)
end

gcm = GaussianCopulaVCModel(gcs)

@info "Initial point:"
init_β!(gcm)
@show gcm.β
fill!(gcm.Σ, 1)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ
# @btime update_Σ!(gcm) setup=(fill!(gcm.Σ, 1))

@show loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)
# @code_warntype loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)
@btime loglikelihood!(gcm.data[1], gcm.β, gcm.τ[1], gcm.Σ, true, false)

@show loglikelihood!(gcm, true, false)
@show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
# @code_warntype loglikelihood!(gcm, false, false)
# @code_llvm loglikelihood!(gcm, false, false)
@btime loglikelihood!(gcm, true, false)

# Solvers:
# Ipopt.IpoptSolver(print_level=0)
# NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000)
# NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000)
# NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)

@info "MLE:"
solver = Ipopt.IpoptSolver(print_level=0)
@time GLMCopula.fit!(gcm, solver)
@show [gcm.β; gcm.τ; gcm.Σ]
@show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
@show loglikelihood!(gcm)
# @btime fit!(gcm, solver) setup=(init_β!(gcm);
#     standardize_res!(gcm); update_quadform!(gcm, true); fill!(gcm.Σ, 1);
#     update_Σ!(gcm))

# Profile.clear()
# @profile begin
for solver in [
    NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000),
    NLopt.NLoptSolver(algorithm=:LN_COBYLA, maxeval=10000),
    NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_CCSAQ, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000),
    NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND_RESTART, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_PRECOND, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON_RESTART, maxeval=4000),
    #NLopt.NLoptSolver(algorithm=:LD_TNEWTON, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_VAR1, maxeval=4000),
    # NLopt.NLoptSolver(algorithm=:LD_VAR2, maxeval=4000),
    # Ipopt.IpoptSolver(print_level=0)
    ]
    println()
    @show solver
    # re-set starting point
    init_β!(gcm)
    fill!(gcm.Σ, 1)
    update_Σ!(gcm)
    # fit
    GLMCopula.fit!(gcm, solver)
    @time GLMCopula.fit!(gcm, solver)
    @show loglikelihood!(gcm)
    @show [gcm.β; gcm.τ; gcm.Σ]
    @show [gcm.∇β; gcm.∇τ; gcm.∇Σ]
    println()
end
# end
# Profile.print(format=:flat)

end
