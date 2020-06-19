
"""
initialize!(gcm, Normal)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least
squares solution.
"""
function initialize_model!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}}) where {T <: BlasReal, D}
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p)
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β s.t gcm.β = inv(cholesky(Symmetric(gcm.XtX)))*xty
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    gcm.τ[1] = gcm.ntotal / rss
    gcm.β
end

function initialize_model!(
    gcm::GLMCopulaVCModel{T, D}) where {T <: BlasReal, D}
    glm_β = glm_regress_model(gcm)[1]
    copyto!(gcm.β, glm_β)
    fill!(gcm.τ, 1.0)
    gcm
end

#to initialize beta for glm

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
#to initialize beta for glm

function glm_regress_model(gcm::Union{GaussianCopulaVCModel{T, D}, GLMCopulaVCModel{T, D}}) where {T <:BlasReal, D}
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
                     obj = obj + loglik_obs(gc.d, gc.y[j], gc.μ[j], 1, 1)
                    end
            end
       if obj > old_obj
         break
       else
         BLAS.axpy!(-1, increment, beta)
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
    return beta
end # function glm_regress

function glm_score_statistic(gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}},
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
function glm_score_statistic(gcm::Union{GaussianCopulaVCModel{T, D}, GLMCopulaVCModel{T, D}},
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
