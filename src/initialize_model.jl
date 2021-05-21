"""
initialize!(gcm, Normal)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least
squares solution.
"""
function initialize_model!(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}) where {T <:BlasReal, D<:Normal}
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

"""
initialize_model!(gcm)

Initialize the linear regression parameters `β` by the weighted least
squares solution.
"""
function initialize_model!(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}) where {T <: BlasReal, D}
    glm_β = glm_regress_model(gcm)
    copyto!(gcm.β, glm_β)
    fill!(gcm.τ, 1.0)
    nothing
end

"""
glm_regress_model(gcm)

Initialize beta for glm model
"""
function glm_regress_model(gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}) where {T <:BlasReal, D}
  (n, p) = gcm.ntotal, gcm.p
   beta = zeros(p)
   ybar = gcm.Ytotal / n
   for iteration = 1:20 # find the intercept by Newton's method
     g1 = GLM.linkinv(gcm.link[1], beta[1]) #  mu
     g2 = GLM.mueta(gcm.link[1], beta[1])  # dmu
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
      fill!(gcm.∇β, 0.0)
      fill!(gcm.Hβ, 0.0)
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
                    obj = obj + GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], 1)
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
      return beta
    else
      old_obj = obj
    end
   end
    # fill!(gcm.∇β, 0.0)
    # fill!(gcm.Hβ, 0.0)
    println("reach here")
    gcm = glm_score_statistic(gcm, beta)
    increment = gcm.Hβ \ gcm.∇β
    BLAS.axpy!(1, increment, beta)
    return beta
end # function glm_regress

"""
glm_regress_model(gcm)

Initialize beta for glm model
"""
function glm_score_statistic(gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}},
   β::Vector{T}) where {T<: BlasReal, D}
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
  return gc
end # function glm_score_statistic

"""
glm_score_statistic(gcm)

Get score
"""
function glm_score_statistic(gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}},
   β::Vector) where {T <: BlasReal, D}
  fill!(gcm.∇β, 0.0)
  fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        gcm.data[i] = glm_score_statistic(gcm.data[i], β)
        gcm.∇β .+= gcm.data[i].∇β
        gcm.Hβ .+= gcm.data[i].Hβ
    end
  return gcm
end # function glm_score_statistic
#
# """
# glm_component_score(gc)
#
# get portion of score that is coming from just the glm
# """
# function glm_component_score(gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}},
#    β::Vector{T}) where {T<: BlasReal, D}
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
# function glm_component_score(gcm::GLMCopulaVCModel{T, D},
#    β::Vector) where {T <: BlasReal, D}
#   fill!(gcm.∇β, 0.0)
#   fill!(gcm.Hβ, 0.0)
#     for i in 1:length(gcm.data)
#         gcm.data[i] = glm_component_score(gcm.data[i], β)
#         gcm.∇β .+= gcm.data[i].∇β
#         gcm.Hβ .+= gcm.data[i].Hβ
#     end
#   return gcm
# end # function glm_score_statistic

# function initialize_glm_object()
#   d = Binomial
#   l = canonicallink(d())
#   Random.seed!(12345)
#   xcov = randn(2)
#   X = zeros(200, 2)
#   for i in 1:200
#     X[i, :] .= xcov
#   end
#   y = []
#   for i in 1:length(Y_nsample)
#     push!(y, vec(Y_nsample[i])...)
#     end
#   return fit(GeneralizedLinearModel, X, y, d(), l)
# end