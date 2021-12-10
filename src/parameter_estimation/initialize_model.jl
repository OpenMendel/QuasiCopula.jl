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

"""
    update_rho!(gcm, empirical_covariance_mat)

Given initial estimates for 'σ2' and 'β', initialize the AR parameter 'ρ' using empirical covariance matrix of Y_1 and Y_2.
"""
function update_rho!(gcm, Y_1, Y_2)
    N = length(gcm.data)
    empirical_covariance_mat = scattermat(hcat(Y_1, Y_2))/N
    n1 = length(gcm.data[1].y)
    ρhat = abs(empirical_covariance_mat[1, 2] / (inv(1 + 0.5 * n1 * gcm.σ2[1]) * sqrt(abs(Statistics.mean(Y_1))) * sqrt(abs(Statistics.mean(Y_2))) * gcm.σ2[1]))
    if ρhat > 1
      copyto!(gcm.ρ, 1.0)
    else
      @inbounds for i in eachindex(gcm.data)
        get_V!(ρhat, gcm.data[i])
      end
      fill!(gcm.Σ, 1.0)
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

function initialize_model!(
    gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    glm_regress_model(gcm)
    @show gcm.β
    fill!(gcm.τ, 1.0)
    println("initializing variance components using MM-Algorithm")
    fill!(gcm.Σ, 1.0)
    update_Σ!(gcm)
    if sum(gcm.Σ) >= 20
      fill!(gcm.Σ, 1.0)
    end
    @show gcm.Σ
    nothing
end

# code inspired from https://github.com/JuliaStats/GLM.jl/blob/master/src/negbinfit.jl
function initialize_model!(
    gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}

    # initial guess for r = 1
    fill!(gcm.r, 1)

    # fit a Poisson regression model to estimate μ, η, β, τ
    println("Initializing NegBin r to Poisson regression values")
    nsample = length(gcm.data)
    gcsPoisson = Vector{GLMCopulaVCObs{T, Poisson{T}, LogLink}}(undef, nsample)
    for (i, gc) in enumerate(gcm.data)
        gcsPoisson[i] = GLMCopulaVCObs(gc.y, gc.X, gc.V, Poisson(), LogLink())
    end
    gcmPoisson = GLMCopulaVCModel(gcsPoisson)
    GLMCopula.fit!(gcmPoisson, IpoptSolver(print_level = 0, max_iter = 10, tol = 10^-2, hessian_approximation = "limited-memory"))

    for i in 1:nsample
        copyto!(gcm.data[i].μ, gcmPoisson.data[i].μ)
        copyto!(gcm.data[i].η, gcmPoisson.data[i].η)
    end
    copyto!(gcm.τ, gcmPoisson.τ)
    copyto!(gcm.β, gcmPoisson.β)

    # update r using maximum likelihood with Newton's method
    for gc in gcm.data
      fill!(gcm.τ, 1.0)
      fill!(gcm.Σ, 1.0)
      fill!(gc.∇β, 0)
      fill!(gc.∇τ, 0)
      fill!(gc.∇Σ, 0)
      fill!(gc.Hβ, 0)
      fill!(gc.Hτ, 0)
      fill!(gc.HΣ, 0)
    end
    update_r!(gcm)

    println("initializing variance components using MM-Algorithm")
    fill!(gcm.Σ, 1)
    update_Σ!(gcm)

    nothing
end

function initialize_model!(gcm::NBCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}

  # initial guess for r = 1
  fill!(gcm.r, 1)
  # data::Vector{NBCopulaARObs{T, D, Link}}
  # println("Ytotal = $(gcm.Ytotal)")
  # println("ntotal = $(gcm.ntotal)")
  # println("p = $(gcm.p)")
  # # parameters
  # println("β = $(gcm.β)")
  # println("τ = $(gcm.τ)")
  # println("ρ = $(gcm.ρ)")
  # println("σ2 = $(gcm.σ2)")
  # println("Σ = $(gcm.Σ)")
  # println("r = $(gcm.r)")
  # println("θ = $(gcm.θ)")
  # working arrays
  # ∇β::Vector{T}   # gradient of beta from all observations
  # ∇ρ::Vector{T}           # gradient of rho from all observations
  # ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
  # ∇r::Vector{T}
  # ∇θ::Vector{T}
  # XtX::Matrix{T}  # X'X = sum_i Xi'Xi
  # Hβ::Matrix{T}    # Hessian from all observations
  # Hρ::Matrix{T}    # Hessian from all observations
  # Hσ2::Matrix{T}    # Hessian from all observations
  # Hr::Matrix{T}
  # Hρσ2::Matrix{T}
  # Hβσ2::Vector{T}
  # Ainv::Matrix{T}
  # Aevec::Matrix{T}
  # M::Matrix{T}
  # vcov::Matrix{T}
  # ψ::Vector{T}
  # # Hβρ::Vector{T}
  # TR::Matrix{T}
  # QF::Matrix{T}         # n-by-1 matrix with qik = res_i' Vi res_i
  # storage_n::Vector{T}
  # storage_m::Vector{T}
  # storage_Σ::Vector{T}
  # d::Vector{D}
  # link::Vector{Link}

  # fit a Poisson regression model to estimate μ, η, β, τ
  println("Initializing NegBin r to Poisson regression values")
  nsample = length(gcm.data)
  gcsPoisson = Vector{GLMCopulaARObs{T, Poisson{T}, LogLink}}(undef, nsample)
  for (i, gc) in enumerate(gcm.data)
      gcsPoisson[i] = GLMCopulaARObs(gc.y, gc.X, Poisson(), LogLink())
  end
  gcmPoisson = GLMCopulaARModel(gcsPoisson)
  optm = GLMCopula.fit!(gcmPoisson, IpoptSolver(print_level = 0, max_iter = 100,
      tol = 10^-3, hessian_approximation = "limited-memory", 
      limited_memory_max_history = 20))
  
  # use poisson regression values of β, μ, η to initialize r, if poisson fit was successful
  if MathProgBase.status(optm) == :Optimal
      for i in 1:nsample
          copyto!(gcm.data[i].μ, gcmPoisson.data[i].μ)
          copyto!(gcm.data[i].η, gcmPoisson.data[i].η)
      end
      copyto!(gcm.β, gcmPoisson.β)

      # update r using maximum likelihood with Newton's method
      for gc in gcm.data
        fill!(gcm.τ, 1.0)
        fill!(gc.∇β, 0)
        fill!(gc.Hβ, 0)
        fill!(gc.varμ, 1)
        fill!(gc.res, 0)
      end
      update_r!(gcm)
  end

  println("initializing variance parameters in AR model using MM-Algorithm")
  fill!(gcm.Σ, 1.0)
  fill!(gcm.ρ, 1.0) # initial guess for rho is 1, because it's a random intercept model?
  update_Σ!(gcm)
  copyto!(gcm.σ2, gcm.Σ)

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
                    if typeof(gc.d) <: NegativeBinomial
                      r = gc.d.r
                      obj = obj + logpdf(D(r, r/(gc.μ[j] + r)), gc.y[j])
                    else
                      obj = obj + GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], 1)
                    end
                   end
           end
      if obj > old_obj
        break
      else
        BLAS.axpy!(-1, increment, gcm.β)
        increment = 0.5 * increment
      end
    end
    # println(iteration," ",old_obj," ",obj," ",steps)
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
  end
