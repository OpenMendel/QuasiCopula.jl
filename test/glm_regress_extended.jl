module glm_regress_comparison

println()
@info "Comparing to GLM regress output"

using Statistics, Distributions, LinearAlgebra, GLM, Test, GLMCopula
using LinearAlgebra: BlasReal, copytri!
#  get score from one obs
# function glm_score_statistic_component(gc::GLMCopulaVCObs{T, D}, β::Vector) where {T <: Real, D}
#   (n, p) = size(gc.X)
#   x = zeros(p)
#   @assert n == length(gc.y)
#   @assert p == length(β)
#   fill!(gc.∇β, 0.0)
#   fill!(gc.Hβ, 0.0)
#   mul!(gc.η, gc.X, β) # z = X * beta
#   update_res!(gc, β)
#   for i = 1:n
#     c = gc.res[i] * gc.w1[i]
#     copyto!(x, gc.X[i, :])
#     BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
#     BLAS.ger!(gc.w2[i], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
#   end
# # increment = gc.Hβ \ gc.∇β
# # score_statistic = dot(gc.∇β, increment)
#   return gc
# end # function glm_score_statistic
#
# #  get score from the full model
# function glm_score_statistic_component(gcm::GLMCopulaVCModel{T, D}, beta) where {T <: Real, D}
#   fill!(gcm.∇β, 0.0)
#   fill!(gcm.Hβ, 0.0)
#     for i in 1:length(gcm.data)
#         gcm.data[i] = glm_score_statistic(gcm.data[i], beta)
#         gcm.∇β .+= gcm.data[i].∇β
#         gcm.Hβ .+= gcm.data[i].Hβ
#     end
#   return gcm
# end # function glm_score_statistic

function glm_score_statistic(gc::GLMCopulaVCObs{T, D},
   β::Vector{T}, Σ::Vector{T}) where {T<: BlasReal, D}
  (n, p) = size(gc.X)
  m = length(gc.V)
  component_score = zeros(p)
  @assert n == length(gc.y)
  @assert p == length(β)
  fill!(gc.∇β, 0.0)
  fill!(gc.Hβ, 0.0)
  mul!(gc.η, gc.X, β) # z = X * beta
  update_res!(gc, β)
  GLMCopula.std_res_differential!(gc)
  for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        # component_score stores ∇resβ*Γ*res (standardized residual)
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, component_score)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    x = zeros(p)
    c = 0.0
    inv1pq = inv(1 + qsum)
    BLAS.syrk!('L', 'N', -abs2(inv1pq), component_score, 1.0, gc.Hβ) # only lower triangular
    for j in 1:length(gc.y)
          c = gc.res[j] #* gc.w1[j]
          copyto!(x, gc.X[j, :])
          BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
          BLAS.axpy!(-inv1pq, component_score, gc.∇β) # first term for each glm score
          BLAS.ger!(gc.w2[j], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
    end
# increment = gc.Hβ \ gc.∇β
# score_statistic = dot(gc.∇β, increment)
  return gc
end # function glm_score_statistic

#  get score from the full model
function glm_score_statistic(gcm::GLMCopulaVCModel{T, D},
   β::Vector) where {T <: BlasReal, D}
  fill!(gcm.∇β, 0.0)
  fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        gcm.data[i] = glm_score_statistic(gcm.data[i], β, gcm.Σ)
        gcm.∇β .+= gcm.data[i].∇β
        gcm.Hβ .+= gcm.data[i].Hβ
    end
  return gcm
end # function glm_score_statistic

function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(ϕ)), y)
#loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)

# to ensure no- infinity from loglikelihood!!
function loglik_obs(::Poisson, y, μ, wt, ϕ)
    y * log(μ) - μ
end

#  glm_regress on one obs
function glm_regress_jl(gc::GLMCopulaVCObs{T, D}, Σ) where {T<: Real, D}
  (n, p) = size(gc.X)
   @assert n == length(gc.y)
   beta = zeros(p)
   (x, z) = (zeros(p), zeros(n))
   ybar = mean(gc.y)
   link = GLM.canonicallink(gc.d)
   for iteration = 1:25 # find the intercept by Newton's method
     g1 = GLM.linkinv(link, beta[1]) #  mu
     g2 = GLM.mueta(link, beta[1])  # dmu
     beta[1] = beta[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
     if abs(g1 - ybar) < 1e-10
       break
     end
   end
   (obj, old_obj, c) = (0.0, 0.0, 0.0)
   epsilon = 1e-8
   for iteration = 1:100 # scoring algorithm
     gc = glm_score_statistic(gc, beta, Σ)
     increment = gc.Hβ \ gc.∇β
     beta = beta + increment
     steps = -1
     fill!(gc.∇β, 0.0)
     for step_halve = 0:3 # step halving
       obj = 0.0
       mul!(gc.η, gc.X, beta) # z = X * beta
       update_res!(gc, beta)
       steps = steps + 1
            gc = glm_score_statistic(gc, beta, Σ)
       for j = 1:n
         obj = obj + loglik_obs(gc.d, gc.y[j], gc.μ[j], 1, 1)
       end
       if obj > old_obj
         break
       else
         beta = beta - increment
         increment = 0.5 * increment
       end
     end
     println(iteration," ",old_obj," ",obj," ",steps)
     if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
       return (beta, obj)
     else
       old_obj = obj
     end
    end
    return (beta, obj)
end # function glm_regress_jl

# glm_regess for all obs in the model object,
#  now for two observations, the same as the above but 2 copies shoule have
# the same beta, and twice the objective
#
function glm_regress_model(gcm::GLMCopulaVCModel{T, D}) where {T <: Real, D}
  (n, p) = gcm.ntotal,  gcm.p
   beta = zeros(p)
   ybar = gcm.Ytotal / n
   link = GLM.canonicallink(gcm.d)
   for iteration = 1:25 # find the intercept by Newton's method
     g1 = GLM.linkinv(link, beta[1]) #  mu
     g2 = GLM.mueta(link, beta[1])  # dmu
     beta[1] =  beta[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
     if abs(g1 - ybar) < 1e-10
       break
     end
   end
   (obj, old_obj, c) = (0.0, 0.0, 0.0)
   epsilon = 1e-8
   for iteration = 1:100 # scoring algorithm
     fill!(gcm.∇β, 0.0)
     fill!(gcm.Hβ, 0.0)
     gcm = glm_score_statistic(gcm, beta)
     increment = gcm.Hβ \ gcm.∇β
     BLAS.axpy!(1, increment, beta)
     steps = -1
     for step_halve = 0:3 # step halving
       obj = 0.0
            for i in 1:length(gcm.data)
                gc = gcm.data[i]
                x = zeros(p)
               mul!(gc.η, gc.X, beta) # z = X * beta
               update_res!(gc, beta)
               steps = steps + 1
                   for j = 1:length(gcm.data[i].y)
                     # c = gc.res[j] * gc.w1[j]
                     # copyto!(x, gc.X[j, :])
                     # BLAS.axpy!(c, x, gcm.∇β) # score = score + c * x
                     obj = obj + loglik_obs(gc.d, gc.y[j], gc.μ[j], 1, 1)
                    end
            end
       if obj > old_obj
         break
       else
         BLAS.axpy!(-1, increment, beta)
         #gcm.β = gcm.β - increment
         increment = 0.5 * increment
       end
     end
     println(iteration," ",old_obj," ",obj," ",steps)
     if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
       return (beta, obj)
     else
       old_obj = obj
     end
    end
    return beta, obj
end # function glm_regress

# Ken's test data
# I want to form the model using the test data Ken has, each obs object will be the same.
function create_gcm_logistic(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());

logistic_β, logistic_logl = glm_regress_jl(logistic_model.data[1], logistic_model.Σ)
@test logistic_β ≈ [ -4.077713431087562
                 1.5046454283733053]
@test logistic_logl ≈ -8.029878464344675

logistic_model.β .= logistic_β
logistic_model = glm_score_statistic(logistic_model, logistic_model.β)
score1a = logistic_model.∇β
@test round(abs2(norm(score1a)), digits  = 15)  == 0


logistic2_β, logistic2_logl = glm_regress_model(logistic_model)
@test logistic2_β ≈ [ -4.077713431087562
                 1.5046454283733053]
@test logistic2_logl ≈ -8.029878464344675*2

# Ken's test data
function create_gcm_poisson(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
             X = reshape(X, 14, 1);
             y = [0., 1 ,2 ,3, 1, 4, 9, 18, 23, 31, 20, 25, 37, 45];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

gcm = create_gcm_poisson(2, Poisson());

poisson_β, poisson_logl = glm_regress_jl(gcm.data[1], gcm.Σ)
@test poisson_β ≈ [0.28503887444394366]
@test poisson_logl ≈ 471.19671943091146

gcm.β .= poisson_β
gcm = glm_score_statistic(gcm, gcm.β)
score2a = gcm.∇β
@test round(abs2(norm(score2a)), digits  = 15)  == 0

poisson2_β, poisson2_logl = glm_regress_model(gcm)
copyto!(gcm.β, poisson2_β)
@test gcm.β ≈ [0.28503887444394366]
@test poisson2_logl ≈ 471.19671943091146*2


##### now what if we add the copula model parts to the score?

function glm_score_statistic2(gc::Union{GLMCopulaVCObs{T, D}, GLMCopulaVCObs{T, D}},
   β::Vector{T}, Σ::Vector{T}) where {T<: BlasReal, D}
  (n, p) = size(gc.X)
  m = length(gc.V)
  component_score = zeros(p)
  @assert n == length(gc.y)
  @assert p == length(β)
  fill!(gc.∇β, 0.0)
  fill!(gc.Hβ, 0.0)
  mul!(gc.η, gc.X, β) # z = X * beta
  update_res!(gc, β)
  GLMCopula.std_res_differential!(gc)
  for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        # component_score stores ∇resβ*Γ*res (standardized residual)
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, component_score)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    x = zeros(p)
    c = 0.0
    inv1pq = inv(1 + qsum)
    BLAS.syrk!('L', 'N', -abs2(inv1pq), component_score, 1.0, gc.Hβ) # only lower triangular
    for j in 1:length(gc.y)
          c = gc.res[j] #* gc.w1[j]
          copyto!(x, gc.X[j, :])
          BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
          BLAS.axpy!(-inv1pq, component_score, gc.∇β) # first term for each glm score
          BLAS.ger!(gc.w2[j], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
    end
# increment = gc.Hβ \ gc.∇β
# score_statistic = dot(gc.∇β, increment)
  return gc
end # function glm_score_statistic

#  get score from the full model
function glm_score_statistic2(gcm::Union{GLMCopulaVCModel{T, D}, GLMCopulaVCModel{T, D}},
   β::Vector) where {T <: BlasReal, D}
  fill!(gcm.∇β, 0.0)
  fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        gcm.data[i] = glm_score_statistic2(gcm.data[i], β, gcm.Σ)
        gcm.∇β .+= gcm.data[i].∇β
        gcm.Hβ .+= gcm.data[i].Hβ
    end
  return gcm
end # function glm_score_statistic

logistic2 = glm_score_statistic2(logistic_model, logistic_model.β)
score1_b = logistic2.∇β
@test round(abs2(norm(score1_b)), digits  = 15)  == 0


gcm2 = glm_score_statistic2(gcm, gcm.β)
score2_b = gcm2.∇β
@test round(abs2(norm(score2_b)), digits  = 15)  == 0
# abs2(norm(score1_b)) = 4.8039767909261363e-26
# abs2(norm(score2_b)) = 1.6983859745046296e-23

end
