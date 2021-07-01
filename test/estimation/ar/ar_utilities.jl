using Random, Test
using LinearAlgebra: BlasReal, copytri!
Random.seed!(1234)
ni = 3
res = rand(3)
Gamma = [2 0.2 0.02; 0.2 2 0.2; 0.02 0.2 2]
Gamma1 = [1 0.1 0.01; 0.1 1 0.1; 0.01 0.1 1]
ρ = 0.1
σ2 = 2
@test transpose(res) * Gamma * res == transpose(res) * Gamma1 * res * σ2

function qf(res, ρ, σ2)
    n = length(res)
    t = 0.0
    for j in 1:n - 1
        t += residualx(res, j, ρ) * σ2^2
    end
    t += σ2 * transpose(res) * res
    t
end

function residualx(res, j, ρ)
    n = length(res)
    term = 0.0
    for k in 1:n-j
        term += res[k] * res[k + j]
    end
    ρ^j * term
end

@test qf(res, ρ, σ2) ≈ transpose(res) * Gamma * res

# this gets r^t * Gamma * r = σ2 *(r^t * V(ρ) * r)

using DataFrames, Random, GLM, GLMCopula, Test
using LinearAlgebra, BenchmarkTools

Random.seed!(1234)

# observations per subject
n = 3
ρ = 0.1
σ2 = 2.0

V = zeros(n, n) # will store the AR(1) structure without sigma2

V = GLMCopula.get_AR_cov(n, ρ, σ2, V)

# true Gamma
Γ = σ2 * V

@test Γ ≈ Gamma 

@test V ≈ Gamma1

dist = Poisson
vecd = [dist(5) for i in 1:n]
N = 100
nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

Y_Nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, N)

Random.seed!(1234)

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, N)

for i in 1:N
    y = Float64.(Y_Nsample[i])
    X = ones(n, 1)
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

gcm = GLMCopulaARModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show exp.(gcm.β);
gc = gcm.data[1]
β  = gcm.β

n_i  = length(gc.y)

gc2 = deepcopy(gc)

function qf(gc, β, ρ, σ2)
    n = length(gc.res)
    q = 0.0
    for j in 1:n - 1
        q += residualx(gc.res, j, ρ, n) * σ2^2
    end
    q += σ2 * dot(gc.res, gc.res)
    0.5 * q
end

function residualx(res, j, ρ, n)
    term = 0.0
    for k in 1:n-j
        term += res[k] * res[k + j]
    end
    ρ^j * term
end

# this is how we do it in the loglikelihood function
function l2!(
    gc::GLMCopulaARObs{T, D, Link},
    β::Vector{T},
    ρ::T,
    σ2::T) where {T <: BlasReal, D, Link}
    n, p = size(gc.X, 1), size(gc.X, 2)
    update_res!(gc, β)
    standardize_res!(gc)
    # form V
    gc.V .= get_AR_cov(n, ρ, σ2, gc.V)

    #evaluate copula loglikelihood
    mul!(gc.storage_n, gc.V, gc.res) # storage_n = V[k] * res
    q = dot(gc.res, gc.storage_n)

    # gradient
    gc.∇ARV .= get_∇ARV(n, ρ, σ2, gc.∇ARV)
    mul!(gc.storage_n, gc.∇ARV, gc.res) # storage_n = ∇ARV * res
    q2 = dot(gc.res, gc.storage_n) # 

    gc.∇2ARV .= get_∇2ARV(n, ρ, σ2, gc.∇2ARV)
    mul!(gc.storage_n, gc.∇2ARV, gc.res) # storage_n = ∇ARV * res
    q3 = dot(gc.res, gc.storage_n) # 

    q, q2, q3
end
# (4.265815533980583, 5.619223300970875, 3.590291262135923)


@test l2!(gc, β, ρ, σ2)[1] ≈ qf(gc, β, ρ, σ2)[1]

d1 = get_∇ARV(n, ρ, σ2, gc.∇ARV)
q2 = 0.5 * σ2 * transpose(gc.res) * d1 * gc.res
@test q2 ≈ 5.619223300970875

hardcoded_q2 = 0.5 * σ2 * 2 * (gc.res[1] * gc.res[2] + gc.res[2] * gc.res[3] + 2 * ρ * gc.res[1] * gc.res[3]) 
@test hardcoded_q2 ≈ q2

function qf2(gc, β, ρ, σ2)
    n = length(gc.res)
    q = 0.0
    for j in 1:n - 1
        q += residualx2(gc.res, j, ρ, n) * σ2^2
    end
    0.5 * q
end

function residualx2(res, j, ρ, n)
    term = 0.0
    for k in 1:n-j
        term += res[k] * res[k + j]
    end
    j * ρ^(j-1) * term
end

@test qf2(gc, β, ρ, σ2) ≈ q2

d2 = get_∇2ARV(n, ρ, σ2, gc.∇2ARV)
q3 = 0.5 * σ2 * transpose(gc.res) * d2 * gc.res
@test q3 ≈ 3.590291262135923

hardcoded_q3 = 0.5 * σ2 * 4 * (gc.res[1] * gc.res[3])
@test hardcoded_q3 ≈ q3

function qf3(gc, β, ρ, σ2)
    n = length(gc.res)
    q = 0.0
    for j in 1:n - 1
        q += residualx3(gc.res, j, ρ, n) * σ2^2
    end
    0.5 * q
end

function residualx3(res, j, ρ, n)
    term = 0.0
    for k in 1:n-j
        term += res[k] * res[k + j]
    end
    j * (j-1) * ρ^(j-2) * term
end

@test qf3(gc, β, ρ, σ2) ≈ q3

# now put it all together 
# this is how we do it in the loglikelihood function
function get_qf!(
    gc::GLMCopulaARObs{T, D, Link},
    β::Vector{T},
    ρ::T,
    σ2::T) where {T <: BlasReal, D, Link}
    n, p = size(gc.X, 1), size(gc.X, 2)
    update_res!(gc, β)
    standardize_res!(gc)
    q = qf(gc, β, ρ, σ2)
    q2 = qf2(gc, β, ρ, σ2)
    q3 = qf3(gc, β, ρ, σ2)
    q, q2, q3
end

@test get_qf!(gc, β, ρ, σ2)[1] ≈ l2!(gc, β, ρ, σ2)[1]
@test get_qf!(gc, β, ρ, σ2)[2] ≈ l2!(gc, β, ρ, σ2)[2]
@test get_qf!(gc, β, ρ, σ2)[3] ≈ l2!(gc, β, ρ, σ2)[3]

@benchmark get_qf!($gc, $β, $ρ, $σ2)

# julia> @benchmark get_qf!($gc, $β, $ρ, $σ2)

# BenchmarkTools.Trial: 
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     160.371 ns (0.00% GC)
#   median time:      161.060 ns (0.00% GC)
#   mean time:        164.000 ns (0.00% GC)
#   maximum time:     517.417 ns (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     784

# julia> 

# julia> @benchmark l2!($gc, $β, $ρ, $σ2)
# BenchmarkTools.Trial: 
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     351.953 ns (0.00% GC)
#   median time:      354.066 ns (0.00% GC)
#   mean time:        366.018 ns (0.00% GC)
#   maximum time:     1.039 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     213


# now put this into loglikelihood!
function logl2!(
    gc::GLMCopulaARObs{T, D, Link},
    β::Vector{T},
    ρ::T,
    σ2::T,
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D, Link}
    n, p = size(gc.X, 1), size(gc.X, 2)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
    end
    needhess && fill!(gc.Hβ, 0)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    standardize_res!(gc)
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # this needs to change ### 
    # # form V
    # gc.V .= get_AR_cov(n, ρ, σ2, gc.V)

    # # we still need storage_n = V[k] * res to store in gc.∇β below
    # mul!(gc.storage_n, gc.V, gc.res) # storage_n = V[k] * res

    if needgrad
        BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    end
    q = qf(gc, β, ρ, σ2)
    # @show q
    c1 = 1 + 0.5 * n * σ2
    c2 = 1 + 0.5 * σ2 * q
    # loglikelihood
    logl = GLMCopula.component_loglikelihood(gc)
    logl += -log(c1)
    logl += log(c2)
    if needgrad
        inv1pq = inv(c2)
        # gradient with respect to rho
        q2 = qf2(gc, β, ρ, σ2)
        # gc.∇ρ .= inv(c2) * 0.5 * σ2 * transpose(gc.res) * gc.∇ARV * gc.res
        gc.∇ρ .= inv(c2) * 0.5 * σ2 * q2

        # gradient with respect to sigma2
        gc.∇σ2 .= -0.5 * n * inv(c1) .+ inv(c2) * 0.5 * q
      if needhess
            q3 = qf3(gc, β, ρ, σ2)
            # hessian for rho
            gc.Hρ .= 0.5 * σ2 * (inv(c2) * q3 - inv(c2)^2 * 0.5 * σ2 * q2^2)
            
            # hessian for sigma2
            gc.Hσ2 .= 0.25 * n^2 * inv(c1)^2 - inv(c2)^2 * (0.25 * q^2)
            
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 0.0, gc.Hβ) # only lower triangular
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            mul!(gc.added_term_numerator, gc.V, gc.∇resβ) # storage_n = V[k] * res
            BLAS.gemm!('T', 'N', σ2, gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
            gc.added_term2 .*= inv1pq
            gc.Hβ .+= gc.added_term2
            gc.Hβ .+= GLMCopula.glm_hessian(gc, β)
      end
      gc.∇β .= gc.∇β .* inv1pq
      gc.∇β .+= GLMCopula.glm_gradient(gc, β, 1.0)
    end
    logl
end

logl2!(gc, β, ρ, σ2)
@benchmark logl2!($gc, $β, $ρ, $σ2)