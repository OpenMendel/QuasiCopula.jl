using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra
using LinearAlgebra: BlasReal, copytri!

Random.seed!(1235)
p = 10   # number of clusters

n = 100  # number of cells

df = DataFrame(gene = repeat('A':'J', outer=n), normal = rand(n * p), nresp = ones(n * p))

#lmm1 = lmm(@formula(nresp ~ 1 + (1|gene)), df);
lmm1 = LinearMixedModel(@formula(nresp ~ 1 +  normal + (1|gene)), df);
# simulate the linear mixed model response and fit it (as a sanity check)

# refit!(simulate!(lmm1, β=[0.2, 1.5], σ=0.01, θ=[5.]))

#simulate the Poisson response
df[!, :counts] = rand.(Poisson.(GLM.linkinv.(canonicallink(Poisson()), response(lmm1))))
df

poisson_formula = @formula(counts ~ 1 + normal + (1|gene));
@time mdl = GeneralizedLinearMixedModel(poisson_formula, df, Poisson());
loglikelihood(mdl)

groups = unique(df[!, :gene])
n, p, m = length(groups), 1, 1
d = Poisson()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = df[!, :gene] .== grp
    ni = count(gidx)
    y = Float64.(df[gidx, :counts])
    normal = Float64.(df[gidx, :normal])
    X = [ones(ni, 1) normal]
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)
GLMCopula.loglikelihood!(gcm, true, true)

gc = gcm.data[1]
β  = gcm.β
Σ  = gcm.Σ
τ  = gcm.τ

@show β
@show Σ
@show τ

n_i  = length(gc.y)

# check if quantities are updated to what i expect them to be theoretically
@test gc.η == gc.X*β                        # systematic linear component
@test gc.μ == exp.(gc.η) # mu = ginverse of XB = mean component for GLM = [exp(eta)]
@test gc.varμ == gc.μ  # variance of the Poisson response as a function of mean mu
@test gc.res ≈ (gc.y - gc.μ)./sqrt.(gc.varμ)# standardized residual for GLM [(y - p)/sqrt(mu)]

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
logl_hard_coded_obs1 = term1 + term2
copula_logl_function = GLMCopula.copula_loglikelihood_addendum(gc, Σ)
@show logl_hard_coded_obs1
@show copula_logl_function
@test copula_logl_function ≈ logl_hard_coded_obs1


#  coming from glm.jl
function poisson_density(y, μ)
    logl = 0.0
    for j in 1:length(y)
        logl += y[j]*log(μ[j]) - μ[j] - log(factorial(y[j]))
    end
    logl
end

term3 = poisson_density(gc.y, gc.μ)

logl_component_logistic = 0.0
logl_component_logistic += component_loglikelihood(gc, τ[1], logl_component_logistic)

@test logl_component_logistic == term3

# full loglikelihood
logl_hard = term1 + term2 + term3

logl_functions = copula_loglikelihood(gc, β, τ[1], Σ)
@test logl_hard == logl_functions

####
# these are slightly off by small decimals
@test gc.dμ ≈ exp.(gc.η)
function poisson_gradient(y, X, dμ, σ2, μ)
    grad = zeros(size(X, 2))
    for j in 1:length(y)
        grad += (y[j] - μ[j])*dμ[j]/σ2[j] * X[j, :]
    end
    grad
end

# check if glm gradient is right
term1_gradient = poisson_gradient(gc.y, gc.X, gc.dμ, gc.varμ, gc.μ)
hardgrad1 = transpose(gc.X)*(gc.y - gc.μ)
term1_grad_fctn = GLMCopula.glm_gradient(gc, β, τ)
@test hardgrad1 ≈ term1_grad_fctn

# find matrix of differentials

# for j = 1 and j = 2, ..., j = end; lets take a look at the first two columns
∇μβ1 = exp.(gc.η[1]) .* gc.X[1, :]
∇μβ2 = exp.(gc.η[2]) .* gc.X[2, :]
# ...
∇μβend = exp.(gc.η[end]) .* gc.X[end, :]

@show ∇μβ1
@show ∇μβ2
@show ∇μβend

∇μη = exp.(gc.η)
@show gc.varμ ≈ ∇μη
@test gc.dμ ≈ gc.varμ
∇ηβ = gc.X

∇μβ = transpose(∇ηβ)*Diagonal(∇μη)

# for j = 1 and j = 2 ,... , j = end; lets take a look at the first two columns
∇σ2β1 = gc.dμ[1] .* gc.X[1, :]
∇σ2β2 = gc.dμ[2] .* gc.X[2, :]
# ...
∇σ2βend = gc.dμ[end] .* gc.X[end, :]

@show ∇σ2β1
@show ∇σ2β2
@show ∇σ2βend

∇σ2β = transpose(gc.X)* Diagonal(gc.dμ)

# for j =1
∇resβ1 = -1/(sqrt(gc.varμ[1])) .* ∇μβ1 - ((1/2gc.varμ[1]) * gc.res[1]) .* ∇σ2β1
∇resβ2 = -1/(sqrt(gc.varμ[2])) .* ∇μβ2 - ((1/2gc.varμ[2]) * gc.res[2]) .* ∇σ2β2
# ...
∇resβend = -1/(sqrt(gc.varμ[end])) .* ∇μβend - ((1/2gc.varμ[end]) * gc.res[end]) .* ∇σ2βend

@show ∇resβ1
@show ∇resβ2
@show ∇resβend

update_res!(gc, β)
standardize_res!(gc)
std_res_differential!(gc)
@test gc.∇resβ[1, :] ≈ ∇resβ1

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

gradient_term2_function = GLMCopula.copula_gradient_addendum(gc, β, τ[1], Σ)
@test gradient_term2 ≈ gradient_term2_function

gradient_hard_code = term1_gradient + gradient_term2
function copula_gradient(gc::GLMCopulaVCObs{T, D}, β, τ, Σ)  where {T<:BlasReal, D}
    fill!(gc.∇β, 0.0)
    gc.∇β .= GLMCopula.glm_gradient(gc, β, τ) .+ GLMCopula.copula_gradient_addendum(gc, β, τ[1], Σ)
end
full_gradient_function = copula_gradient(gc, β, τ, Σ)

@test full_gradient_function ≈ gradient_hard_code

@show loglikelihood!(gcm, true, true)
@show loglikelihood(mdl)


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

function poisson_density2(β::Vector)
    η = gc.X*β                        # systematic linear component
    μ = exp.(η)   # mu = ginverse of XB = mean component for GLM = [p]
    dμ = exp.(η)
    varμ = dμ
    logl = sum(gc.y .* log.(μ) .- μ)
end

logl_term3 = poisson_density2(β)
@show logl_term3

g = x -> ForwardDiff.gradient(poisson_density2, x)

gradientmagictest = g(β)
@show gradientmagictest

@test term1_grad_fctn ≈ gradientmagictest

h1 = x -> ForwardDiff.hessian(poisson_density2, x)
hessian1magictest = h1(β)
@show hessian1magictest

function standardized_residual_firstobs(β::Vector)
    η = gc.X*β                        # systematic linear component
    μ = exp.(η) # mu = ginverse of XB = mean component for GLM = [p]
    varμ = exp.(η)
    res = (gc.y[1] - μ[1]) / sqrt(varμ[1])
end

g2 = x -> ForwardDiff.gradient(standardized_residual_firstobs, x)
gradientmagictest2 = g2(β)
@show gradientmagictest2

@test gc.∇resβ[1, :] ≈ gradientmagictest2


function copula_loglikelihood_addendum1(β::Vector)
  m = length(gc.V)
  η = gc.X*β                        # systematic linear component
  μ = exp.(η) # mu = ginverse of XB = mean component for GLM = [p]
  varμ = exp.(η)
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
    logl = poisson_density2(β) + copula_loglikelihood_addendum1(β)
    logl
end

@show full_loglikelihood(β)
@show norm(logl_functions - full_loglikelihood(β))

g4 = x -> ForwardDiff.gradient(full_loglikelihood, x)

gradientmagictest4 = g4(β)
@show gradientmagictest4

full_gradient_function = copula_gradient(gc, β, τ, Σ)
@test full_gradient_function ≈ gradientmagictest4

h = x -> ForwardDiff.hessian(full_loglikelihood, x)
hessianmagictest = h(β)
@show hessianmagictest

@show loglikelihood!(gcm, true, true)
@show gcm.∇β

# takes only 22 iterations
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, derivative_test = "first-order", hessian_approximation = "limited-memory"))

@show loglikelihood!(gcm, true, true)
@show gcm.∇β

gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β
fill!(gcm.Σ, 1.0)
update_Σ!(gcm)
GLMCopula.loglikelihood!(gcm, true, true)

# as is in the paper it takes 39 iterations and then using the other term we get 35 iterations
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
GLMCopula.loglikelihood!(gcm, true, true)
# -1863.1043633723516

# 0.218887 seconds (1.08 M allocations: 101.188 MiB, 18.91% gc time)
# using MixedModels -1891.2586221674821
