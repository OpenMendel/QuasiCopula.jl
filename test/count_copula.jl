# module Poisson_Logistic_Test

# println()
# @info "Modeling Counts using GLM.jl"
# using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Test, GLMCopula
# using LinearAlgebra: BlasReal, copytri!
# @reexport using Ipopt
# @reexport using NLopt

using Statistics, Distributions, LinearAlgebra, GLM, Test

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM
using LinearAlgebra: BlasReal, copytri!
using Ipopt
using NLopt


#  form data object to store data and intermediate fields
struct glm_VCobs{T <: BlasReal, D}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β
    xtx::Matrix{T}  # Xi'Xi
    #xtw2x::Matrix{T}# Xi'W2iXi where W2i = Diagonal(mueta(link, Xi*B))^2/Var(mu_i)
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    storage_n::Vector{T} # storage_n = V_i[k] * res_i
    storage_p::Vector{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function glm_VCobs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D
    ) where {T <: BlasReal, D}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, p, n)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    xtx = transpose(X) * X
    #xtw2x = Matrix{T}(undef, p, n)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    glm_VCobs{T, D}(y, X, V, ∇β, ∇resβ, ∇Σ, Hβ, xtx, res,
      t, q, storage_n, storage_p, η, μ, varμ, dμ, d, w1, w2)
end

#  form model object to store data and intermediate fields
struct glm_VCModel{T <: BlasReal, D}  <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{glm_VCobs{T, D}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    # ∇τ::Vector{T}
    ∇Σ::Vector{T}
    Hβ::Matrix{T}    # Hessian from all observations
    # Hτ::Matrix{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    # XtW2X::Matrix{T} # X'W2X = sum_i Xi'W2iXi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::D
end

function glm_VCModel(gcs::Vector{glm_VCobs{T, D}}) where {T <: BlasReal, D}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    XtX = zeros(T, p, p) # sum_i xi'xi
    # XtW2X = zeros(T, p, p)
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        #BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    glm_VCModel{T, D}(gcs, Ytotal, ntotal, p, m, β, Σ,
        ∇β, ∇Σ, Hβ, XtX, TR, QF,
        storage_n, storage_m, storage_Σ, gcs[1].d)
end

"""
update_res!(gc, β)

Update the residual vector according to `β` and the canonical inverse link to the given distribution.
"""
function update_res!(
    gc::glm_VCobs{T, D},
    β::Vector{T}
    ) where {T <: BlasReal, D}
    mul!(gc.η, gc.X, β)
    for i in 1:length(gc.y)
        gc.μ[i] = GLM.linkinv(canonicallink(gc.d), gc.η[i])
        gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i])
        gc.dμ[i] = GLM.mueta(canonicallink(gc.d), gc.η[i])
        gc.w1[i] = gc.dμ[i] / gc.varμ[i]
        gc.w2[i] = gc.dμ[i]^2 / gc.varμ[i]
        gc.res[i] = gc.y[i] - gc.μ[i]
    end
    return gc.res
end


function standardize_res!(
    gc::glm_VCobs{T, D}
    ) where {T <: BlasReal, D}
    for j in 1:length(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end



#  get score from one obs


#  get score from one obs
function glm_score_statistic(pc::glm_VCobs{T, D}, β::Vector) where {T <: Real, D}
  (n, p) = size(pc.X)
  x = zeros(p)
  @assert n == length(pc.y)
  @assert p == length(β)
  fill!(pc.∇β, 0.0)
  fill!(pc.Hβ, 0.0)
  mul!(pc.η, pc.X, β) # z = X * beta
  update_res!(pc, β)
  for i = 1:n
    c = pc.res[i] * pc.w1[i]
    copyto!(x, pc.X[i, :])
    BLAS.axpy!(c, x, pc.∇β) # pc.∇β = pc.∇β + r_ij(β) * mueta* x
    BLAS.ger!(pc.w2[i], x, x, pc.Hβ) # pc.Hβ = pc.Hβ + r_ij(β) * x * x'
  end
# increment = pc.Hβ \ pc.∇β
# score_statistic = dot(pc.∇β, increment)
  return pc
end # function glm_score_statistic

#  get score from the full model
function glm_score_statistic(pcm::glm_VCModel{T, D}, beta) where {T <: Real, D}
  fill!(pcm.∇β, 0.0)
  fill!(pcm.Hβ, 0.0)
    for i in 1:length(pcm.data)
        pcm.data[i] = glm_score_statistic(pcm.data[i], beta)
        pcm.∇β .+= pcm.data[i].∇β
        pcm.Hβ .+= pcm.data[i].Hβ
    end
  return pcm
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
function glm_regress_jl(pc::glm_VCobs{T, D}) where {T<: Real, D}
  (n, p) = size(pc.X)
   @assert n == length(pc.y)
   beta = zeros(p)
   (x, z) = (zeros(p), zeros(n))
   ybar = mean(pc.y)
   link = GLM.canonicallink(pc.d)
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
     pc = glm_score_statistic(pc, beta)
     increment = pc.Hβ \ pc.∇β
     beta = beta + increment
     steps = -1
     fill!(pc.∇β, 0.0)
     for step_halve = 0:3 # step halving
       obj = 0.0
       mul!(pc.η, pc.X, beta) # z = X * beta
       update_res!(pc, beta)
       steps = steps + 1
            pc = glm_score_statistic(pc, beta)
       for j = 1:n
         obj = obj + loglik_obs(pc.d, pc.y[j], pc.μ[j], 1, 1)
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
function glm_regress_model(pcm::glm_VCModel{T, D}) where {T <: Real, D}
  (n, p) = pcm.ntotal,  pcm.p
   beta = zeros(p)
   ybar = pcm.Ytotal / n
   link = GLM.canonicallink(pcm.d)
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
     fill!(pcm.∇β, 0.0)
     fill!(pcm.Hβ, 0.0)
     pcm = glm_score_statistic(pcm, beta)
     increment = pcm.Hβ \ pcm.∇β
     BLAS.axpy!(1, increment, beta)
     steps = -1
     for step_halve = 0:3 # step halving
       obj = 0.0
            for i in 1:length(pcm.data)
                pc = pcm.data[i]
                x = zeros(p)
               mul!(pc.η, pc.X, beta) # z = X * beta
               update_res!(pc, beta)
               steps = steps + 1
                   for j = 1:length(pcm.data[i].y)
                     c = pc.res[j] * pc.w1[j]
                     copyto!(x, pc.X[j, :])
                     BLAS.axpy!(c, x, pcm.∇β) # score = score + c * x
                     obj = obj + loglik_obs(pc.d, pc.y[j], pc.μ[j], 1, 1)
                    end
            end
       if obj > old_obj
         break
       else
         BLAS.axpy!(-1, increment, beta)
         #pcm.β = pcm.β - increment
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

### Now that we know we can get the beta lets try to update Σ now with MM

"""
update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::glm_VCobs{T, D}) where {T<:BlasReal, D}
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

function update_Σ_jensen!(
    gcm::glm_VCModel{T, D},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        standardize_res!(gcm.data[i])
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q # now QF is formed from the standardized residuals
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = sum_k^m qi[k] sigmai_[k] # denom of numerator
        gcm.storage_n .= inv.(1 .+ gcm.storage_n) # 1/ (1 + sum_k^m qi[k] sigmai_[k]) # denom of numerator
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n) # store numerator = b_i / (1 + a b_i)
        gcm.Σ .*= gcm.storage_m # multiply
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.Σ ./= gcm.storage_m
        # monotonicity diagnosis
        verbose && println(sum(log, 1 .+ (gcm.QF * gcm.Σ)) -
            sum(log, 1 .+ gcm.TR * gcm.Σ))
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
            verbose && println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

"""
update_Σ!(gc)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
update_Σ! = update_Σ_jensen!


@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)

# term1=  102.95885585666409
#logl = 471.1967187139634
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.Σ;



"""
    loglikelihood!(gc::glm_VCobs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
"""

function loglikelihood!(
    gc::glm_VCobs{T, D},
    β::Vector{T},
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    update_res!(gc, β)
    standardize_res!(gc)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇Σ, 0)
        fill!(gc.∇resβ, 0.0)
        std_res_differential!(gc)
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t)
    logl = - log(1 + tsum)
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # ∇β stores X'*W*Γ*res (standardized residual)
            BLAS.gemv!('N', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)

    # sum up the component loglikelihood
    for j = 1:length(gc.y)
        logl += loglik_obs(gc.d, gc.y[j], gc.μ[j], 1.0, 1.0)
    end
    # gradient
    if needgrad
        x = zeros(p)
        c = 0.0
        inv1pq = inv(1 + qsum)
        if needhess
            for j = 1:length(gc.y) # get the hessian and the score
              c = gc.res[j] * gc.w1[j]
              copyto!(x, gc.X[j, :])
              BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
              BLAS.ger!(gc.w2[j], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
            end
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
        end
        BLAS.gemv!('N', 1.0, gc.∇resβ, gc.res, -inv1pq, gc.∇β)
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
    end
    # output
    logl
end
#


function std_res_differential!(gc)
    for j in 1:length(gc.y)
        gc.∇resβ[:, j] = -inv(sqrt(gc.varμ[j]))*gc.dμ[j].*transpose(gc.X[j, :]) - (1/2gc.dμ[j])*gc.res[j] * gc.dμ[j].*transpose(gc.X[j, :])
    end
    gc
end

#-102.95885585666409 + 8.646784118270125 = -94.31207173839397 for obs 14

function loglikelihood!(
    gcm::glm_VCModel{T, D},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇Σ, 0)
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.Σ, needgrad, needhess)
        #println(logl)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    needhess && (gcm.Hβ)
    logl
end

function fit!(
    gcm::glm_VCModel{T, D},
    solver=NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000)
    ) where {T<:BlasReal, D}
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill(Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm)
    gcm
end

function MathProgBase.initialize(
    gcm::glm_VCModel{T, D},
    requested_features::Vector{Symbol}) where {T<:BlasReal, D}
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

function copy_par!(
    gcm::glm_VCModel{T, D},
    par::Vector) where {T<:BlasReal, D}
    copyto!(gcm.β, par)
    par
end

MathProgBase.features_available(gcm::glm_VCModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::glm_VCModel{T, D},
    par::Vector)  where {T<:BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate loglikelihood
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::glm_VCModel{T, D},
    grad::Vector,
    par::Vector)  where {T<:BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    nothing
end

function MathProgBase.hesslag_structure(gcm::glm_VCModel{T, D})  where {T<:BlasReal, D}
    Iidx = Vector{Int}(undef, (gcm.p * (gcm.p + 1)) >> 1)
    Jidx = similar(Iidx)
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            Iidx[ct] = i
            Jidx[ct] = j
            ct += 1
        end
    end
    Iidx, Jidx
end

function MathProgBase.eval_hesslag(
    gcm::glm_VCModel{T, D},
    H::Vector{T},
    par::Vector{T},
    σ::T) where {T <: BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate Hessian
    loglikelihood!(gcm, true, true)
    # copy Hessian elements into H
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            H[ct] = gcm.Hβ[i, j]
            ct += 1
        end
    end
    H .*= σ
end
@test poisson_β ≈ [0.28503887444394366]
@test poisson_logl ≈ 471.19671943091146
@test loglikelihood!(gcm, true, false) ≈ 471.19671943091146*2
@show gcm.∇β
# @show gcm.∇τ
@show gcm.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time fit!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.Σ
@show loglikelihood!(gcm, true, false) #≈ -163.35545251
@show gcm.∇β
@show gcm.∇Σ

#
# end
