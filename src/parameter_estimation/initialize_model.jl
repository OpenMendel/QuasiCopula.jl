"""
    initialize!(gcm{D}) where D<: Normal

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least
squares solution for the Normal distribution.
"""
function initialize_model!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}}) where {T <:BlasReal, D<:Normal, Link}
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
    update_rho!(gcm, empirical_covariance_mat)

Given initial estimates for 'σ2' and 'β', initialize the AR parameter 'ρ' using empirical covariance matrix of Y_1 and Y_2.
"""
function update_rho!(gcm, Y_1, Y_2)
    N = length(gcm.data)
    empirical_covariance_mat = scattermat(hcat(Y_1, Y_2))/N
    n1 = length(gcm.data[1].y)
    ρhat = abs(empirical_covariance_mat[1, 2] /(inv(1 + 0.5 * n1 * gcm.σ2[1]) * sqrt(Statistics.mean(Y_1)) * sqrt(Statistics.mean(Y_2)) * gcm.σ2[1]))
    if ρhat > 1
      copyto!(gcm.ρ, 1.0)
    else
      @inbounds for i in eachindex(gcm.data)
        get_V!(ρhat, gcm.data[i])
      end
      update_Σ!(gcm)
        if gcm.Σ[1] < 10
          copyto!(gcm.σ2, gcm.Σ)
        else
          copyto!(gcm.σ2, 1.0)
        end
      copyto!(gcm.ρ, ρhat)
    end
    nothing
end

"""
    initialize_model!(gcm{D}) where D<: Poisson, Bernoulli

Initialize the linear regression parameters `β` by the weighted least
squares solution.
"""
function initialize_model!(
    gcm::GLMCopulaARModel{T, D}) where {T <: BlasReal, D}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    glm_regress_model(gcm)
    fill!(gcm.τ, 1.0)
    fill!(gcm.ρ, 1.0)
    fill!(gcm.Σ, 1.0)
    update_Σ!(gcm)
    copyto!(gcm.σ2, gcm.Σ)
    nothing
end

function initialize_model!(
  gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
  println("initializing β using Newton's Algorithm under Independence Assumption")
  glm_regress_model(gcm)
  fill!(gcm.τ, 1.0)
  println("initializing variance components using MM-Algorithm")
  fill!(gcm.Σ, 1.0)
  update_Σ!(gcm)
  nothing
end

"""
    glm_regress_model(gcm)

Initialize beta for glm model.
"""
function glm_regress_model(gcm::Union{GLMCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}, NBCopulaVCModel{T, D, Link}})  where {T <:BlasReal, D, Link}
  (n, p) = gcm.ntotal, gcm.p
   fill!(gcm.β, 0.0)
   ybar = gcm.Ytotal / n
   for iteration = 1:20 # find the intercept by Newton's method
     g1 = GLM.linkinv(gcm.link[1], gcm.β[1]) #  mu
     g2 = GLM.mueta(gcm.link[1], gcm.β[1])  # dmu
     gcm.β[1] =  gcm.β[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
     if abs(g1 - ybar) < 1e-10
       break
     end
   end
   (obj, old_obj, c) = (0.0, 0.0, 0.0)
   epsilon = 1e-8
   for iteration = 1:100 # scoring algorithm
    fill!(gcm.∇β, 0.0)
    fill!(gcm.Hβ, 0.0)
    gcm = glm_score_statistic(gcm)
    increment = gcm.Hβ \ gcm.∇β
    BLAS.axpy!(1, increment, gcm.β)
    steps = -1
    for step_halve = 0:3 # step halving
      obj = 0.0
      fill!(gcm.∇β, 0.0)
      fill!(gcm.Hβ, 0.0)
           for i in 1:length(gcm.data)
               gc = gcm.data[i]
               x = zeros(p)
              update_res!(gc, gcm.β)
              steps = steps + 1
                  for j = 1:length(gc.y)
                    c = gc.res[j] * gc.w1[j]
                    copyto!(x, gc.X[j, :])
                    BLAS.axpy!(c, x, gcm.∇β) # score = score + c * x
                    obj = obj + GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], 1)
                   end
           end
      if obj > old_obj
        break
      else
        BLAS.axpy!(-1, increment, gcm.β)
        increment = 0.5 * increment
      end
    end
    println(iteration," ",old_obj," ",obj," ",steps)
    if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
      return gcm.β
    else
      old_obj = obj
    end
   end
    gcm = glm_score_statistic(gcm)
    increment = gcm.Hβ \ gcm.∇β
    BLAS.axpy!(1, increment, gcm.β)
    return gcm.β
end # function glm_regress

"""
glm_score_statistic(gc, β, τ)

Get gradient and hessian of beta to for a single independent vector of observations.
"""
function glm_score_statistic(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, NBCopulaVCObs{T, D, Link}},
  β::Vector{T}, τ::T) where {T<: BlasReal, D, Link}
   fill!(gc.∇β, 0.0)
   fill!(gc.Hβ, 0.0)
   update_res!(gc, β)
   gc.∇β .= glm_gradient(gc, β, τ)
   gc.Hβ .= GLMCopula.glm_hessian(gc, β)
   gc
end 

"""
glm_score_statistic(gcm)

Get gradient and hessian of beta to do newtons method on independent glm model for all observations in gcm model object.
"""
function glm_score_statistic(gcm::Union{GLMCopulaVCModel{T, D}, GLMCopulaARModel{T, D}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
  fill!(gcm.∇β, 0.0)
  fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        gcm.data[i] = glm_score_statistic(gcm.data[i], gcm.β, gcm.τ[1])
        gcm.∇β .+= gcm.data[i].∇β
        gcm.Hβ .+= gcm.data[i].Hβ
    end
  return gcm
end # function glm_score_statistic