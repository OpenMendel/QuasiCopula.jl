module glm_regress_comparison

println()
@info "Comparing to GLM regress output"

using Statistics, Distributions, LinearAlgebra, GLM, Test, GLMCopula
#  get score from one obs
function glm_score_statistic(gc::glm_VCobs{T, D}, β::Vector) where {T <: Real, D}
  (n, p) = size(gc.X)
  x = zeros(p)
  @assert n == length(gc.y)
  @assert p == length(β)
  fill!(gc.∇β, 0.0)
  fill!(gc.Hβ, 0.0)
  mul!(gc.η, gc.X, β) # z = X * beta
  update_res!(gc, β)
  for i = 1:n
    c = gc.res[i] * gc.w1[i]
    copyto!(x, gc.X[i, :])
    BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
    BLAS.ger!(gc.w2[i], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
  end
# increment = gc.Hβ \ gc.∇β
# score_statistic = dot(gc.∇β, increment)
  return gc
end # function glm_score_statistic

#  get score from the full model
function glm_score_statistic(gcm::glm_VCModel{T, D}, beta) where {T <: Real, D}
  fill!(gcm.∇β, 0.0)
  fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        gcm.data[i] = glm_score_statistic(gcm.data[i], beta)
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

# function loglik_obs(::Poisson, y, μ, wt, ϕ)
#     p = pdf(Poisson(μ), y)
#     p = map(y -> y == 0.0 ? 6.2101364865661445e-176 : y, p)
#     wt*log(p)
# end

#  glm_regress on one obs
function glm_regress_jl(gc::glm_VCobs{T, D}) where {T<: Real, D}
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
     gc = glm_score_statistic(gc, beta)
     increment = gc.Hβ \ gc.∇β
     beta = beta + increment
     steps = -1
     fill!(gc.∇β, 0.0)
     for step_halve = 0:3 # step halving
       obj = 0.0
       mul!(gc.η, gc.X, beta) # z = X * beta
       update_res!(gc, beta)
       steps = steps + 1
            gc = glm_score_statistic(gc, beta)
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
function glm_regress_model(gcm::glm_VCModel{T, D}) where {T <: Real, D}
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
                     c = gc.res[j] * gc.w1[j]
                     copyto!(x, gc.X[j, :])
                     BLAS.axpy!(c, x, gcm.∇β) # score = score + c * x
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
           gcs = Vector{glm_VCobs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = glm_VCobs(y, X, V, dist)
           end
           gcms = glm_VCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());

logistic_β, logistic_logl = glm_regress_jl(logistic_model.data[1])
@test logistic_β ≈ [ -4.077713431087562
                 1.5046454283733053]
@test logistic_logl ≈ -8.029878464344675

logistic2_β, logistic2_logl = glm_regress_model(logistic_model)
@test logistic2_β ≈ [ -4.077713431087562
                 1.5046454283733053]
@test logistic2_logl ≈ -8.029878464344675*2

# Ken's test data
function create_gcm_poisson(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{glm_VCobs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
             X = reshape(X, 14, 1);
             y = [0., 1 ,2 ,3, 1, 4, 9, 18, 23, 31, 20, 25, 37, 45];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = glm_VCobs(y, X, V, dist)
           end
           gcms = glm_VCModel(Vector(gcs))
           return gcms
       end

gcm = create_gcm_poisson(2, Poisson());

poisson_β, poisson_logl = glm_regress_jl(gcm.data[1])
@test poisson_β ≈ [0.28503887444394366]
@test poisson_logl ≈ 471.19671943091146

poisson2_β, poisson2_logl = glm_regress_model(gcm)
copyto!(gcm.β, poisson2_β)
@test gcm.β ≈ [0.28503887444394366]
@test poisson2_logl ≈ 471.19671943091146*2

end
