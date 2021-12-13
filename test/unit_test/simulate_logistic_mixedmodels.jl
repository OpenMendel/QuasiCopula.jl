using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra
using LinearAlgebra: BlasReal, copytri!

Random.seed!(1235)
p = 10   # number of genes

n = 100  # number of cells

df = DataFrame(gene = repeat('A':'J', outer=n), normal = rand(n * p), nresp = ones(n * p))

#lmm1 = lmm(@formula(nresp ~ 1 + (1|gene)), df);
lmm1 = LinearMixedModel(@formula(nresp ~ 1 +  normal + (1|gene)), df);
# simulate the linear mixed model response and fit it (as a sanity check)

# refit!(simulate!(lmm1, β=[0.2, 1.5], σ=0.01, θ=[5.]))

#simulate the Logistic response
df[!, :counts] = rand.(Bernoulli.(GLM.linkinv.(canonicallink(Bernoulli()), response(lmm1))))
df

glmm1 = MixedModels.fit!(GeneralizedLinearMixedModel(@formula(counts ~ 1 + normal + (1|gene)), df, Bernoulli()))
loglikelihood(glmm1)
# glmm_β = [0.828822; 0.0270186]
# glmm_Σ = 0.0
# logl_glmm = -611.7027

groups = unique(df[!, :gene])
n, p, m = length(groups), 1, 1
d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = df[!, :gene] .== grp
    ni = count(gidx)
    y = Float64.(df[gidx, :counts])
    normal = Float64.(df[gidx, :normal])
    X = [ones(ni, 1) normal]
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
GLMCopula.loglikelihood!(gcm, true, true)

gc = gcm.data[1]
β  = gcm.β
Σ  = gcm.Σ
τ  = gcm.τ
update_res!(gc, β)
standardize_res!(gc)
@show β
@show Σ
@show τ

n_i  = length(gc.y)

# check if quantities are updated to what i expect them to be theoretically
@test gc.η == gc.X*β                        # systematic linear component
@test gc.μ == exp.(gc.η)./(1 .+ exp.(gc.η)) # mu = ginverse of XB = mean component for GLM = [p]
@test gc.varμ == gc.μ .*(1 .- gc.μ)         # variance of the Bernoulli response as a function of mean mu [p(1-p)]
@test gc.res ≈ (gc.y - gc.μ)./sqrt.(gc.varμ)# standardized residual for GLM [(y - p)/sqrt(p(1-p))]


# part of loglikelihood specific to our density
# term 1:
trace_gamma = Σ[1]*tr(gc.V[1])
@test trace_gamma ≈ n_i*Σ[1]

trace_gamma_half = trace_gamma/2
@show trace_gamma_half

term1 = -log(1 + trace_gamma_half)
@show term1;

# term 2:
quad_form_standardized_res_half = (Σ[1]*transpose(gc.res)*gc.V[1]*gc.res)/2
@show quad_form_standardized_res_half

term2 = log(1 + quad_form_standardized_res_half)
@show term2

"""
    copula_loglikelihood_addendum!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the parts of the loglikelihood that is particular to our density for a single observation. These parts are an addendum to the component loglikelihood from the GLM density.
"""
function copula_loglikelihood_addendum(gc::GLMCopulaVCObs{T, D, Link}, Σ::Vector{T}) where {T<: BlasReal, D, Link}
  m = length(gc.V)
  for k in 1:m
    mul!(gc.storage_n, gc.V[k], gc.res)
    gc.q[k] = dot(gc.res, gc.storage_n) / 2
  end
  tsum = dot(Σ, gc.t)
  logl = -log(1 + tsum)
  qsum  = dot(Σ, gc.q)
  inv1pq = inv(1 + qsum)
  logl += log(1 + qsum)
  logl
end

logl_hard_coded_obs1 = term1 + term2
copula_logl_function = copula_loglikelihood_addendum(gc, Σ)
@show logl_hard_coded_obs1
@show copula_logl_function
@test copula_logl_function ≈ logl_hard_coded_obs1


#  coming from glm.jl
function logistic_density(y, μ)
    logl = 0.0
    for j in 1:length(y)
        logl += y[j]*log(μ[j]) + (1 - y[j])*log(1 - μ[j])
    end
    logl
end

term3 = logistic_density(gc.y, gc.μ)

logl_component_logistic = component_loglikelihood(gc)


@test logl_component_logistic == term3

"""
    copula_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link})
Calculates the full loglikelihood for our copula model for a single observation
"""
function copula_loglikelihood(gc::GLMCopulaVCObs{T, D, Link}, β::Vector{T}, τ::T, Σ::Vector{T}) where {T<: BlasReal, D, Link}
#first get the loglikelihood from the component density with glm.jl
  logl = 0.0
  update_res!(gc, β)
  if gc.d == Normal()
    σinv = sqrt(τ[1])# general variance
    standardize_res!(gc, σinv)
  else
    standardize_res!(gc)
  end
  logl += copula_loglikelihood_addendum(gc, Σ)
  logl += GLMCopula.component_loglikelihood(gc)
  logl
end

# full loglikelihood
logl_hard = term1 + term2 + term3

logl_functions = copula_loglikelihood(gc, β, τ[1], Σ)
@test logl_hard == logl_functions


# these are slightly off by small decimals
@test gc.dμ ≈ exp.(gc.η)./(1 .+ exp.(gc.η)).^2
function logistic_gradient(y, X, dμ, σ2, μ)
    grad = zeros(size(X, 2))
    for j in 1:length(y)
        grad += (y[j] - μ[j])*dμ[j]/σ2[j] * X[j, :]
    end
    grad
end

# check if glm gradient is right
term1_gradient = logistic_gradient(gc.y, gc.X, gc.dμ, gc.varμ, gc.μ)
hardgrad1 = transpose(gc.X)*(gc.y - gc.μ)
term1_grad_fctn = GLMCopula.glm_gradient(gc, β, τ)
@test hardgrad1 ≈ term1_grad_fctn

# find matrix of differentials

# for j = 1 and j = 2, ..., j = end; lets take a look at the first two columns
∇μβ1 = exp.(gc.η[1]) / (1 + exp.(gc.η[1]))^2 .* gc.X[1, :]
∇μβ2 = exp.(gc.η[2]) / (1 + exp.(gc.η[2]))^2 .* gc.X[2, :]
# ...
∇μβend = exp.(gc.η[end]) / (1 + exp.(gc.η[end]))^2 .* gc.X[end, :]

@show ∇μβ1
@show ∇μβ2
@show ∇μβend

∇μη = exp.(gc.η) ./ (1 .+ exp.(gc.η)).^2
@show gc.varμ ≈ ∇μη
@test gc.dμ ≈ gc.varμ
∇ηβ = gc.X

∇μβ = transpose(∇ηβ)*Diagonal(∇μη)

# for j = 1 and j = 2 ,... , j = end; lets take a look at the first two columns
∇σ2β1 = (1 - 2*gc.μ[1]) * gc.dμ[1] .* gc.X[1, :]
∇σ2β2 = (1 - 2*gc.μ[2]) * gc.dμ[2] .* gc.X[2, :]
# ...
∇σ2βend = (1 - 2*gc.μ[end]) * gc.dμ[end] .* gc.X[end, :]

@show ∇σ2β1
@show ∇σ2β2
@show ∇σ2βend

∇σ2β = transpose(gc.X)* Diagonal((1 .- 2*gc.μ) .* gc.dμ)

# for j =1
∇resβ1 = -1/(sqrt(gc.varμ[1])) .* ∇μβ1 - ((1/2gc.varμ[1]) * gc.res[1]) .* ∇σ2β1
∇resβ2 = -1/(sqrt(gc.varμ[2])) .* ∇μβ2 - ((1/2gc.varμ[2]) * gc.res[2]) .* ∇σ2β2
# ...
∇resβend = -1/(sqrt(gc.varμ[end])) .* ∇μβend - ((1/2gc.varμ[end]) * gc.res[end]) .* ∇σ2βend

update_res!(gc, β)
standardize_res!(gc)
std_res_differential!(gc)
@test gc.∇resβ[1, :] ≈ -1/(sqrt(gc.varμ[1])) .* ∇μβ1 - ((1/2gc.varμ[1]) * gc.res[1]) .* ∇σ2β1

# gradient of components specific to copula density
Γ1 = Σ[1]*gc.V[1]

grad_t2_numerator = transpose(gc.∇resβ) * Γ1 * gc.res       # new term ∇resβ^t * Γ * res
@show grad_t2_numerator

quadratic_form_half = (transpose(gc.res) * Γ1 * gc.res)/2
@show quadratic_form_half
@test quadratic_form_half ≈ quad_form_standardized_res_half # from the loglikelihood 'qsum'

grad_t2_denominator = inv(1 + quadratic_form_half)
@show grad_t2_denominator

gradient_term2 = grad_t2_numerator * grad_t2_denominator

"""
    copula_gradient_addendum(gc)
Compute the part of gradient specific to copula density with respect to beta for a single observation
"""
function copula_gradient_addendum(
    gc::GLMCopulaVCObs{T, D, Link},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D, Link}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    if gc.d  ==  Normal()
            sqrtτ = sqrt.(τ[1])
            standardize_res!(gc, sqrtτ)
        else
            sqrtτ = 1.0
            standardize_res!(gc)
        end
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # evaluate copula loglikelihood
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end

    qsum  = dot(Σ, gc.q)
    # gradient
        denom = 1 .+ qsum
        inv1pq = inv(denom) #0.9625492359318475
        # component_score = W1i(Yi -μi)
        gc.storage_p2 .= gc.∇β .* inv1pq
        gc.storage_p2 .*= sqrtτ # since we already standardized it above
        gc.storage_p2
end

gradient_term2_function = copula_gradient_addendum(gc, β, τ[1], Σ)
@test gradient_term2 ≈ gradient_term2_function

gradient_hard_code = term1_gradient + gradient_term2
function copula_gradient(gc::GLMCopulaVCObs{T, D}, β, τ, Σ)  where {T<:BlasReal, D}
    fill!(gc.∇β, 0.0)
    gc.∇β .= GLMCopula.glm_gradient(gc, β, τ) .+ copula_gradient_addendum(gc, β, τ[1], Σ)
end
full_gradient_function = copula_gradient(gc, β, τ, Σ)

@test full_gradient_function ≈ gradient_hard_code

#### now check the matrix calculus using ForwardDiff.jl

gc = gcm.data[1]
β  = gcm.β
Σ  = gcm.Σ
τ  = gcm.τ

@show β
@show Σ
@show τ

n_i  = length(gc.y)

update_res!(gc, β)

function logistic_density2(β::Vector)
    η = gc.X*β                        # systematic linear component
    μ = exp.(η)./(1 .+ exp.(η)) # mu = ginverse of XB = mean component for GLM = [p]
    dμ = exp.(η)./(1 .+ exp.(η)).^2
    varμ = dμ
    logl = sum(gc.y .* log.(μ) .+ (1 .- gc.y).*log.(1 .- μ))
end

logl_term3 = logistic_density2(β)
@show logl_term3

g = x -> ForwardDiff.gradient(logistic_density2, x)

gradientmagictest = g(β)
@show gradientmagictest

@test term1_grad_fctn ≈ gradientmagictest

h1 = x -> ForwardDiff.hessian(logistic_density2, x)
hessian1magictest = h1(β)
@show hessian1magictest

function standardized_residual_firstobs(β::Vector)
    η = gc.X*β                        # systematic linear component
    μ = exp.(η)./(1 .+ exp.(η)) # mu = ginverse of XB = mean component for GLM = [p]
    varμ = exp.(η)./(1 .+ exp.(η)).^2
    res = (gc.y[1] - μ[1]) / sqrt(varμ[1])
end

g2 = x -> ForwardDiff.gradient(standardized_residual_firstobs, x)
gradientmagictest2 = g2(β)
@show gradientmagictest2

@test gc.∇resβ[1, :] ≈ gradientmagictest2


function copula_loglikelihood_addendum1(β::Vector)
  m = length(gc.V)
  η = gc.X*β                        # systematic linear component
  μ = exp.(η)./(1 .+ exp.(η)) # mu = ginverse of XB = mean component for GLM = [p]
  varμ = exp.(η)./(1 .+ exp.(η)).^2
  res = (gc.y .- μ) ./ sqrt.(varμ)
  trace_gamma = Σ[1]*tr(gc.V[1])
  trace_gamma_half = trace_gamma/2

  term1 = -log(1 + trace_gamma_half) # -1.252762968495368
  quad_form_standardized_res_half = (Σ[1]*transpose(res)*gc.V[1]*res)/2
  term2 = log(1 + quad_form_standardized_res_half) # 0.0381700599136237
  logl_hard_coded_obs1 = term1 + term2
  logl_hard_coded_obs1
end

g3 = x -> ForwardDiff.gradient(copula_loglikelihood_addendum1, x)

@show copula_loglikelihood_addendum1(β)

gradientmagictest3 = g3(β)
@show gradientmagictest3

@test gradient_term2_function ≈ gradientmagictest3

h2 = x -> ForwardDiff.hessian(copula_loglikelihood_addendum1, x)
hessian2magictest = h2(β)
@show hessian2magictest


function full_loglikelihood(β::Vector)
    logl = 0.0
    logl = logistic_density2(β) + copula_loglikelihood_addendum1(β)
    logl
end

@show full_loglikelihood(β)
@show norm(logl_functions - full_loglikelihood(β))

g4 = x -> ForwardDiff.gradient(full_loglikelihood, x)

gradientmagictest4 = g4(β)
@show gradientmagictest4

function copula_gradient(gc::GLMCopulaVCObs{T, D}, β, τ, Σ)  where {T<:BlasReal, D}
    fill!(gc.∇β, 0.0)
    gc.∇β .= GLMCopula.glm_gradient(gc, β, τ) .+ copula_gradient_addendum(gc, β, τ[1], Σ)
end

full_gradient_function = copula_gradient(gc, β, τ, Σ)
@test full_gradient_function ≈ gradientmagictest4

h = x -> ForwardDiff.hessian(full_loglikelihood, x)
hessianmagictest = h(β)
@show hessianmagictest

@show loglikelihood!(gcm, true, true)
# -611.7033220576704

# using fitold.jl
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, mehrotra_algorithm="yes", warm_start_init_point="yes", hessian_approximation = "exact"))
# ourlogl using MM-Algorithm only # -611.7026900495172 (6 iterations and 0.5 seconds)

# using fitnew.jl since the variance component is roughly 0, we get some numerical issues