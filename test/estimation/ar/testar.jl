using DataFrames, Random, GLM, GLMCopula, Test, ToeplitzMatrices
using LinearAlgebra, BenchmarkTools

using LinearAlgebra: BlasReal, copytri!

Random.seed!(1234)

# sample size
N = 10000
# observations per subject
n = 50
ρ = 0.9
σ2 = 0.1

V = zeros(n, n) # will store the AR(1) structure without sigma2

mean = 5

dist = Poisson

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = vec[i-1] * ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end
# V = get_AR_cov(n, ρ, σ2, V)
V = get_V(ρ, n)

# true Gamma
Γ = σ2 * V

vecd = [dist(mean) for i in 1:n]
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
@show exp.(gcm.β)
@show gcm.ρ
@show gcm.σ2

### now sigma2 is initialized now we need rho
Y_1 = [Y_Nsample[i][1] for i in 1:N]
Y_2 = [Y_Nsample[i][2] for i in 1:N]

update_rho!(gcm, Y_1, Y_2)
@show exp.(gcm.β)
@show gcm.ρ
@show gcm.σ2

loglikelihood!(gcm, true, true)

gcm1 = deepcopy(gcm);
gcm2 = deepcopy(gcm);
gcm3 = deepcopy(gcm);
gcm4 = deepcopy(gcm);
gcm5 = deepcopy(gcm);

### Quasi-Newton ### 
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, hessian_approximation = "limited-memory"))
# @time GLMCopula.fit!(gcm1, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive", hessian_approximation = "limited-memory"))
# this one is fastest for quasi-newton
@time GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive",  mu_oracle = "loqo", hessian_approximation = "limited-memory"))

### Newton ### 
# @time GLMCopula.fit!(gcm3, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, hessian_approximation = "exact"))
# this one is fastest for newton
@time GLMCopula.fit!(gcm4, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive", hessian_approximation = "exact"))
# @time GLMCopula.fit!(gcm5, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive",  mu_oracle = "loqo", hessian_approximation = "exact"))

# @show gcm.∇θ
@show gcm2.∇θ
# @show gcm3.∇θ
@show gcm4.∇θ
# @show gcm5.∇θ
# @show gcm.θ
@show gcm2.θ
# @show gcm3.θ
@show gcm4.θ
# @show gcm5.θ

using BenchmarkTools
@benchmark loglikelihood!($gcm2, true, true)


# # # # # ### for one obs
# gcm = GLMCopulaARModel(gcs);

# initialize_model!(gcm)
# @show gcm.β
# @show exp.(gcm.β);
# fill!(gcm.ρ, ρ)
# fill!(gcm.σ2, σ2)
# gcm2 = deepcopy(gcm);

# # for one obs 
# gc = gcm.data[1]
# n_i  = length(gc.y)

# gc2 = deepcopy(gc)

# ## these are the V, V' and V'' 
# # observations per subject
# V = zeros(n_i, n_i) # will store the AR(1) structure without sigma2

# """
#     get_AR_cov(n, ρ, σ2, V)
# Forms the AR(1) covariance structure given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# """
# function get_AR_cov(n, ρ, σ2, V)
#     fill!(V, 1.0)
#     @inbounds for i in 1:n
#         @inbounds for j in i+1:n
#             power = j - i
#             @inbounds for k in 1:power
#                 V[i, j] *= ρ
#             end
#             V[j, i] = V[i, j]
#         end
#     end
#     V
# end

# """
#     get_∇ARV(n, ρ, σ2, V)
# Forms the first derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# """
# function get_∇ARV(n, ρ, σ2, ∇ARV)
#     @inbounds for i in 1:n
#         ∇ARV[i, i] = 0.0
#         @inbounds for j in i+1:n
#             power = j - i - 1
#             ∇ARV[i, j] = (j-i)
#             @inbounds for k in 1:power
#                 ∇ARV[i, j] *= ρ
#             end
#             ∇ARV[j, i] = ∇ARV[i, j]
#         end
#     end
#     ∇ARV
# end

# """
#     get_∇A2RV(n, ρ, σ2, V)
# Forms the second derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# """
# function get_∇2ARV(n, ρ, σ2, ∇2ARV)
#     @inbounds for i in 1:n
#         ∇2ARV[i, i] = 0.0
#         @inbounds for j in i+1:n
#             pw = (j-i-1)
#             if pw == 0
#                 ∇2ARV[i, j] = 0.0
#             else
#                 ∇2ARV[i, j] = (pw + 1)*(pw)
#                 @inbounds for k in 1:pw - 1
#                     ∇2ARV[i, j] *= ρ
#                 end
#             end
#             ∇2ARV[j, i] = ∇2ARV[i, j]
#         end
#     end
#     ∇2ARV
# end

# V = GLMCopula.get_AR_cov(n_i, ρ, σ2, V)

# get_V!(ρ, gc)
# @test gc.V == V

# # @benchmark GLMCopula.get_AR_cov($n_i, $ρ, $σ2, $gc.V)

# Vprime = zeros(n_i, n_i)
# Vprime = GLMCopula.get_∇ARV(n_i, ρ, σ2, Vprime)


# function get_∇V!(ρ, gc)
#     gc.vec[1] = 0.0
#     gc.vec[2] = 1.0
#     for i in 3:Integer(gc.n)
#         gc.vec[i] = (i-1) * inv(i-2) * gc.vec[i-1] * ρ
#     end
#     gc.∇ARV .= ToeplitzMatrices.SymmetricToeplitz(gc.vec)
#     nothing
# end


# # @benchmark get_∇ARV($n_i, $ρ, $σ2, $gc.∇ARV)

# Vprime2 = zeros(n_i, n_i)
# Vprime2 = GLMCopula.get_∇2ARV(n_i, ρ, σ2, Vprime2)

# # @benchmark get_∇2ARV($n_i, $ρ, $σ2, $gc.∇2ARV)

# """
#     get_AR_cov(n, ρ, σ2, V)
# Forms the AR(1) covariance structure given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# """
# function get_AR_cov2(n, ρ, σ2, V)
#     fill!(V, 1.0)
#     @inbounds for i in 1:n
#         @inbounds for j in i+1:n
#             power = j - i
#             for k in 1:power
#                 V[i, j] *= ρ
#             end
#             V[j, i] = V[i, j]
#         end
#     end
#     V
# end

# V2 = zeros(n_i, n_i)
# V2 = get_AR_cov2(n_i, ρ, σ2, V2)
# @test V2 ≈ V
# @benchmark get_AR_cov2($n_i, $ρ, $σ2, $gc.V)

# # """
# #     get_∇ARV(n, ρ, σ2, V)
# # Forms the first derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# # """
# function get_∇ARV2(n, ρ, σ2, ∇ARV)
#     @inbounds for i in 1:n
#         ∇ARV[i, i] = 0.0
#         @inbounds for j in i+1:n
#             power = j - i - 1
#             ∇ARV[i, j] = (j-i)
#             for k in 1:power
#                 ∇ARV[i, j] *= ρ
#             end
#             ∇ARV[j, i] = ∇ARV[i, j]
#         end
#     end
#     ∇ARV
# end

# Vp2 = zeros(n_i, n_i)
# Vp2 = get_∇ARV2(n_i, ρ, σ2, Vp2)
# @test Vp2 ≈ Vprime
# @benchmark get_∇ARV2($n_i, $ρ, $σ2, $gc.V)


# # """
# #     get_∇A2RV(n, ρ, σ2, V)
# # Forms the second derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
# # """
# function get_∇2ARV3(n, ρ, σ2, ∇2ARV)
#     @inbounds for i in 1:n
#         ∇2ARV[i, i] = 0.0
#         @inbounds for j in i+1:n
#             pw = (j-i-1)
#             if pw == 0
#                 ∇2ARV[i, j] = 0.0
#             else
#                 ∇2ARV[i, j] = (pw + 1)*(pw)
#                 for k in 1:pw - 1
#                     ∇2ARV[i, j] *= ρ
#                 end
#             end
#             ∇2ARV[j, i] = ∇2ARV[i, j]
#         end
#     end
#     ∇2ARV
# end
# Vp3 = zeros(n_i, n_i)
# Vp3 = get_∇2ARV3(n_i, ρ, σ2, Vp3)
# @test Vp3 ≈ Vprime2
# @benchmark get_∇2ARV3($n_i, $ρ, $σ2, $gc.V)
