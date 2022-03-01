"""
    initialize_model!(gcm)

Initialize the linear regression parameters `β` using Newton's Algorithm under Independence Assumption, update variance components using MM-Algorithm.
"""
function initialize_model!(
    gcm::GLMCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    fill!(gcm.τ, 1.0)
    fill!(gcm.θ, 1.0)
    println("initializing variance components using MM-Algorithm")
    update_θ!(gcm)
    copyto!(gcm.σ2, gcm.θ)
    # println("initializing ρ using method of moments")
    copyto!(gcm.ρ, 0.2)
    # update_sigma_rho!(gcm)
    nothing
end

function initialize_model!(
    gcm::GLMCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    println("initializing σ2 and ρ using method of moments")
    # update_sigma_rho!(gcm)
    copyto!(gcm.ρ, 0.2)
    nothing
end

function offdiag(A::AbstractMatrix)
    [A[ι] for ι in CartesianIndices(A) if ι[1] ≠ ι[2]]
end

"""
    update_sigma_rho!(gcm)

Given initial estimates for 'β', initialize the correlation parameter 'ρ' and 'σ2' using empirical variance covariance matrix of Y_1 and Y_2.
"""
function update_sigma_rho!(gcm::GLMCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}
    N = length(gcm.data)
    di = length(gcm.data[1].y)
    Y = zeros(N, di)
    @inbounds for j in 1:di
        Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
    end
    empirical_covariance_mat = scattermat(Y) ./ N
    VarY_k = maximum(Diagonal(empirical_covariance_mat))
    CovY_kY_l = mean(GLMCopula.offdiag(empirical_covariance_mat))

    update_res!(gcm)
    # theoretical variance
    σ2_k = zeros(N)
    @inbounds for i in eachindex(gcm.data)
        σ2_k[i] = mean(gcm.data[i].varμ)
    end
    σ2_k_mean = mean(σ2_k)

    σ2hat = 2 * (VarY_k - σ2_k_mean) * inv((1 + σ2_k_mean * 3) - σ2_k_mean)
    ρhat = CovY_kY_l * inv(σ2_k_mean * σ2hat)
    @show ρhat
    @show σ2hat
    if σ2hat > 1
        @inbounds for i in eachindex(gcm.data)
            get_V!(gcm.ρ[1], gcm.data[i])
        end
        fill!(gcm.θ, 1.0)
        update_θ!(gcm)
        copyto!(gcm.σ2, gcm.θ[1])
    else
        copyto!(gcm.σ2, σ2hat)
    end
    if ρhat > 1
        copyto!(gcm.ρ, 0.25)
    elseif ρhat < -1
        copyto!(gcm.ρ, -0.25)
    else
        copyto!(gcm.ρ, ρhat)
    end
    nothing
end

"""
    update_sigma_rho!(gcm)

Given initial estimates for 'β', initialize the correlation parameter 'ρ' and 'σ2' using empirical variance covariance matrix of Y_1 and Y_2.
"""
function update_sigma_rho!(gcm::NBCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("method of moments for NB CS")
    N = length(gcm.data)
    di = length(gcm.data[1].y)
    Y = zeros(N, di)
    @inbounds for j in 1:di
        Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
    end
    corY = StatsBase.cor(Y)
    empirical_correlation_mean = mean(GLMCopula.offdiag(corY))
    ρhat = empirical_correlation_mean * ( 1 + 0.5 * di + kurtosis(Y))
    @show ρhat
    if ρhat > 1
        copyto!(gcm.ρ, 0.5)
    elseif ρhat < -1
        copyto!(gcm.ρ, -0.1)
    else
        copyto!(gcm.ρ, ρhat)
    end
    @show gcm.σ2
    nothing
end

function update_sigma_rho!(gcm::Union{GLMCopulaARModel{T, D, Link}, NBCopulaARModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    N = length(gcm.data)
    di = length(gcm.data[1].y)
    Y = zeros(N, di)
    @inbounds for j in 1:di
        Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
    end
    empirical_covariance_mat = scattermat(Y) ./ N

    update_res!(gcm)
    # theoretical variance
    σ2_k = zeros(N)
    @inbounds for i in eachindex(gcm.data)
        σ2_k[i] = mean(gcm.data[i].varμ)
    end
    σ2_k_mean = mean(σ2_k)

    ρhat = abs((empirical_covariance_mat[1, di] * (1 + 0.5 * di * gcm.σ2[1]) / σ2_k_mean)/gcm.σ2[1])^(1/(di - 1))
    @show ρhat
    if ρhat < 0
        copyto!(gcm.ρ, 0.1)
    elseif ρhat > 1
        copyto!(gcm.ρ, 0.5)
    else
        @inbounds for i in eachindex(gcm.data)
            get_V!(ρhat, gcm.data[i])
        end
        fill!(gcm.θ, σ2_k_mean)
        update_θ!(gcm)
        @show gcm.θ
        if gcm.θ[1] > 1
            copyto!(gcm.σ2, 1.0)
        else
            copyto!(gcm.σ2, gcm.θ)
        end
        copyto!(gcm.ρ, ρhat)
    end
    nothing
end

"""
    initialize_model!(gcm{GLMCopulaVCModel, Poisson_Bernoulli_VCModel, NBCopulaVCModel})

Initialize the linear regression parameters `β` using GLM.jl, and update variance components using MM-Algorithm.
"""
function initialize_model!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}}) where {T <: BlasReal, D, Link,  VD, VL}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    @show gcm.β
    fill!(gcm.τ, 1.0)
    println("initializing variance components using MM-Algorithm")
    fill!(gcm.θ, 1.0)
    update_θ!(gcm)
    if sum(gcm.θ) >= 20
      fill!(gcm.θ, 1.0)
    end
    @show gcm.θ
    nothing
end

# """
#     initialize_model!(gcm{NBCopulaVCModel})
#
# Initialize the linear regression parameters `β` using GLM.jl, and update variance components using MM-Algorithm.
# """
# function initialize_model!(
#     gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
#     println("initializing β using GLM.jl")
#     initialize_beta!(gcm)
#     @show gcm.β
#     println("initializing r using Newton update")
#     fill!(gcm.r, 1)
#     GLMCopula.update_r!(gcm)
#     fill!(gcm.τ, 1.0)
#     println("initializing variance components using MM-Algorithm")
#     fill!(gcm.θ, 1.0)
#     update_θ!(gcm)
#     if sum(gcm.θ) >= 20
#       fill!(gcm.θ, 1.0)
#     end
#     @show gcm.θ
#     nothing
# end

"""
    initialize_beta!(gcm{Poisson_Bernoulli_VCModel})

Initialize the linear regression parameters `β` using GLM.jl
"""
function initialize_beta!(gcm::Poisson_Bernoulli_VCModel{T, VD, VL}) where {T <: BlasReal, VD, VL}
    # form df
    Xstack = []
    Y1stack = zeros(length(gcm.data))
    Y2stack = zeros(length(gcm.data))
    @inbounds for i in 1:length(gcm.data)
        push!(Xstack, gcm.data[i].X[1, 1:Integer((gcm.p)/2)])
        Y1stack[i] = gcm.data[i].y[1]
        Y2stack[i] = gcm.data[i].y[2]
    end
    X = vcat(transpose(Xstack)...)

    poisson_glm = GLM.glm(X, Y1stack, gcm.vecd[1][1], gcm.veclink[1][1])
    bernoulli_glm = GLM.glm(X, Y2stack, gcm.vecd[1][2], gcm.veclink[1][2])
    copyto!(gcm.β, [poisson_glm.pp.beta0; bernoulli_glm.pp.beta0])
    nothing
end

"""
    initialize_beta!(gcm{GLMCopulaVCModel})

Initialize the linear regression parameters `β` using GLM.jl
"""
function initialize_beta!(gcm::Union{GLMCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}, GLMCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    # form df
    Xstack = []
    Ystack = []
    @inbounds for i in 1:length(gcm.data)
        push!(Xstack, gcm.data[i].X)
        push!(Ystack, gcm.data[i].y)
    end
    Xstack = [vcat(Xstack...)][1]
    Ystack = [vcat(Ystack...)][1]
    fit_glm = GLM.glm(Xstack, Ystack, gcm.d[1], gcm.link[1])
    copyto!(gcm.β, fit_glm.pp.beta0)
    nothing
end

# # code inspired from https://github.com/JuliaStats/GLM.jl/blob/master/src/negbinfit.jl
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
    optm = GLMCopula.fit!(gcmPoisson, IpoptSolver(print_level = 0,
        max_iter = 100, tol = 10^-2, hessian_approximation = "limited-memory"))

    # use poisson regression values of β, μ, η to initialize r, if poisson fit was successful
    if MathProgBase.status(optm) == :Optimal
        for i in 1:nsample
            copyto!(gcm.data[i].μ, gcmPoisson.data[i].μ)
            copyto!(gcm.data[i].η, gcmPoisson.data[i].η)
        end
        copyto!(gcm.τ, gcmPoisson.τ)
        copyto!(gcm.β, gcmPoisson.β)

        # update r using maximum likelihood with Newton's method
        for gc in gcm.data
            fill!(gcm.τ, 1.0)
            fill!(gcm.θ, 1.0)
            fill!(gc.∇β, 0)
            fill!(gc.∇τ, 0)
            fill!(gc.∇θ, 0)
            fill!(gc.Hβ, 0)
            fill!(gc.Hτ, 0)
            fill!(gc.Hθ, 0)
        end
        println("initializing r using Newton update")
        GLMCopula.update_r!(gcm)
    else
        fill!(gcm.τ, 1)
        fill!(gcm.β, 0)
    end

    println("initializing variance components using MM-Algorithm")
    fill!(gcm.θ, 1)
    update_θ!(gcm)

    nothing
end

function initialize_model!(gcm::NBCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}

  # initial guess for r = 1
  fill!(gcm.r, 1)

  # fit a Poisson regression model to estimate μ, η, β, τ
  println("Initializing NegBin r to Poisson regression values")
  nsample = length(gcm.data)
  gcsPoisson = Vector{GLMCopulaCSObs{T, Poisson{T}, LogLink}}(undef, nsample)
  for (i, gc) in enumerate(gcm.data)
      gcsPoisson[i] = GLMCopulaCSObs(gc.y, gc.X, Poisson(), LogLink())
  end
  gcmPoisson = GLMCopulaCSModel(gcsPoisson)
  initialize_model!(gcmPoisson)
  copyto!(gcm.β, gcmPoisson.β)
  copyto!(gcm.σ2, gcmPoisson.σ2)

      # update r using maximum likelihood with Newton's method
      for gc in gcm.data
          fill!(gcm.τ, 1.0)
          fill!(gc.∇β, 0)
          fill!(gc.Hβ, 0)
          fill!(gc.varμ, 1)
          fill!(gc.res, 0)
      end
      println("initializing r using Newton update")
      GLMCopula.update_r!(gcm)

  println("initializing variance parameters in CS model using mom")
  # update_sigma_rho!(gcm)
  copyto!(gcm.ρ, 0.2)
  nothing
end

function initialize_model!(gcm::NBCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}

  # initial guess for r = 1
  fill!(gcm.r, 1)

  # fit a Poisson regression model to estimate μ, η, β, τ
  println("Initializing NegBin r to Poisson regression values")
  nsample = length(gcm.data)
  gcsPoisson = Vector{GLMCopulaARObs{T, Poisson{T}, LogLink}}(undef, nsample)
  for (i, gc) in enumerate(gcm.data)
      gcsPoisson[i] = GLMCopulaARObs(gc.y, gc.X, Poisson(), LogLink())
  end
  gcmPoisson = GLMCopulaARModel(gcsPoisson)
  initialize_model!(gcmPoisson)
  copyto!(gcm.β, gcmPoisson.β)
  copyto!(gcm.σ2, gcmPoisson.σ2)

      # update r using maximum likelihood with Newton's method
      for gc in gcm.data
          fill!(gcm.τ, 1.0)
          fill!(gc.∇β, 0)
          fill!(gc.Hβ, 0)
          fill!(gc.varμ, 1)
          fill!(gc.res, 0)
      end
      println("initializing r using Newton update")
      GLMCopula.update_r!(gcm)

  println("initializing variance parameters in CS model using mom")
  # update_sigma_rho!(gcm)
  copyto!(gcm.ρ, 0.2)
  nothing
end

# """
#     glm_regress_model(gcm)
#
# Initialize beta for glm model.
# """
# function glm_regress_model(gcm::Union{GLMCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}, NBCopulaVCModel{T, D, Link}})  where {T <:BlasReal, D, Link}
#   (n, p) = gcm.ntotal, gcm.p
#    fill!(gcm.β, 0.0)
#    ybar = gcm.Ytotal / n
#    for iteration = 1:20 # find the intercept by Newton's method
#      g1 = GLM.linkinv(gcm.link[1], gcm.β[1]) #  mu
#      g2 = GLM.mueta(gcm.link[1], gcm.β[1])  # dmu
#      gcm.β[1] =  gcm.β[1] - clamp((g1 - ybar) / g2, -1.0, 1.0)
#      if abs(g1 - ybar) < 1e-10
#        break
#      end
#    end
#    (obj, old_obj, c) = (0.0, 0.0, 0.0)
#    epsilon = 1e-8
#    for iteration = 1:100 # scoring algorithm
#     fill!(gcm.∇β, 0.0)
#     fill!(gcm.Hβ, 0.0)
#     gcm = glm_score_statistic(gcm)
#     increment = gcm.Hβ \ gcm.∇β
#     BLAS.axpy!(1, increment, gcm.β)
#     steps = -1
#     for step_halve = 0:3 # step halving
#       obj = 0.0
#       fill!(gcm.∇β, 0.0)
#       fill!(gcm.Hβ, 0.0)
#            for i in 1:length(gcm.data)
#                gc = gcm.data[i]
#                x = zeros(p)
#               update_res!(gc, gcm.β)
#               steps = steps + 1
#                   for j = 1:length(gc.y)
#                     c = gc.res[j] * gc.w1[j]
#                     copyto!(x, gc.X[j, :])
#                     BLAS.axpy!(c, x, gcm.∇β) # score = score + c * x
#                     if typeof(gc.d) <: NegativeBinomial
#                       r = gc.d.r
#                       obj = obj + logpdf(D(r, r/(gc.μ[j] + r)), gc.y[j])
#                     else
#                       obj = obj + GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], 1)
#                     end
#                    end
#            end
#       if obj > old_obj
#         break
#       else
#         BLAS.axpy!(-1, increment, gcm.β)
#         increment = 0.5 * increment
#       end
#     end
#     # println(iteration," ",old_obj," ",obj," ",steps)
#     if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
#       return gcm.β
#     else
#       old_obj = obj
#     end
#    end
#     gcm = glm_score_statistic(gcm)
#     increment = gcm.Hβ \ gcm.∇β
#     BLAS.axpy!(1, increment, gcm.β)
#     return gcm.β
# end # function glm_regress
#
# """
#     glm_regress_model(gcm)
#
# Initialize beta for glm model for the poisson and bernoulli mixed distribution.
# """
# function glm_regress_model(gcm::Poisson_Bernoulli_VCModel{T, VD, VL})  where {T <:BlasReal, VD, VL}
#   (n, p) = gcm.ntotal, gcm.p
#    fill!(gcm.β, 0.0)
#    y1bar = gcm.Y1total / n
#    y2bar = gcm.Y2total / n
#    ybar = [y1bar; y2bar]
#    veclink = gcm.veclink[1]
#    for k in 1:2 # each of the distributions
#        for iteration = 1:20 # find the intercept by Newton's method
#          g1 = GLM.linkinv(veclink[k], gcm.β[Integer(p - (p / k) + 1)]) #  mu
#          g2 = GLM.mueta(veclink[k], gcm.β[Integer(p - (p / k) + 1)])  # dmu
#          gcm.β[Integer(p - (p / k) + 1)] =  gcm.β[Integer(p - (p / k) + 1)] - clamp((g1 - ybar[k]) / g2, -1.0, 1.0)
#          if abs(g1 - ybar[k]) < 1e-10
#            break
#          end
#        end
#    end
#    (obj, old_obj, c) = (0.0, 0.0, 0.0)
#    epsilon = 1e-8
#    for iteration = 1:100 # scoring algorithm
#     fill!(gcm.∇β, 0.0)
#     fill!(gcm.Hβ, 0.0)
#     gcm = glm_score_statistic(gcm)
#     increment = gcm.Hβ \ gcm.∇β
#     BLAS.axpy!(1, increment, gcm.β)
#     steps = -1
#     for step_halve = 0:3 # step halving
#       obj = 0.0
#       fill!(gcm.∇β, 0.0)
#       fill!(gcm.Hβ, 0.0)
#            for i in 1:length(gcm.data)
#                gc = gcm.data[i]
#                x = zeros(p)
#               update_res!(gc, gcm.β)
#               steps = steps + 1
#                   for j = 1:length(gc.y)
#                     c = gc.res[j] * gc.w1[j]
#                     copyto!(x, gc.X[j, :])
#                     BLAS.axpy!(c, x, gcm.∇β) # score = score + c * x
#                     obj = obj + GLMCopula.loglik_obs(gc.vecd[j], gc.y[j], gc.μ[j], gc.wt[j], 1)
#                    end
#            end
#       if obj > old_obj
#         break
#       else
#         BLAS.axpy!(-1, increment, gcm.β)
#         increment = 0.5 * increment
#       end
#     end
#     # println(iteration," ",old_obj," ",obj," ",steps)
#     if iteration > 1 && abs(obj - old_obj) < epsilon * (abs(old_obj) + 1.0)
#       return gcm.β
#     else
#       old_obj = obj
#     end
#    end
#     gcm = glm_score_statistic(gcm)
#     increment = gcm.Hβ \ gcm.∇β
#     BLAS.axpy!(1, increment, gcm.β)
#     return gcm.β
# end # function glm_regress
#
# """
# glm_score_statistic(gc, β, τ)
#
# Get gradient and hessian of beta to for a single independent vector of observations.
# """
# function glm_score_statistic(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, Poisson_Bernoulli_VCObs{T, VD, VL}},
#   β::Vector{T}, τ::T) where {T<: BlasReal, D, Link, VD, VL}
#    fill!(gc.∇β, 0.0)
#    fill!(gc.Hβ, 0.0)
#    update_res!(gc, β)
#    gc.∇β .= glm_gradient(gc)
#    gc.Hβ .= GLMCopula.glm_hessian(gc)
#    gc
# end
#
# """
# glm_score_statistic(gcm)
#
# Get gradient and hessian of beta to do newtons method on independent glm model for all observations in gcm model object.
# """
# function glm_score_statistic(gcm::Union{GLMCopulaVCModel{T, D}, GLMCopulaARModel{T, D}, NBCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}}) where {T <: BlasReal, D, Link, VD, VL}
#   fill!(gcm.∇β, 0.0)
#   fill!(gcm.Hβ, 0.0)
#     for i in 1:length(gcm.data)
#         gcm.data[i] = glm_score_statistic(gcm.data[i], gcm.β, gcm.τ[1])
#         gcm.∇β .+= gcm.data[i].∇β
#         gcm.Hβ .+= gcm.data[i].Hβ
#     end
#   return gcm
#   end
#
