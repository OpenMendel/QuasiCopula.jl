println()
@info "Testing Logistic Regression: the independent case confirms independent but the verbAgg data is acting weird on a small subset"

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Statistics, GLMCopula, Test
using LinearAlgebra: BlasReal, copytri!

# Ken's test data
# I want to form the model using the test data Ken has, each obs object will be the same, and we expect the estimated Sigma to be 0.0 (independent)
function create_gcm_logistic(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0.0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());
initialize_model!(logistic_model)

fill!(logistic_model.Σ, 1)
update_Σ!(logistic_model)
loglikelihood!(logistic_model, true, false)

mod_test = glm_score_statistic(logistic_model, logistic_model.β)

# logistic_β = [ -4.077713431087562
#                  1.5046454283733053]

@test loglikelihood!(logistic_model, true, false) ≈  -8.029878464344675*2
@show logistic_model.∇β
@show logistic_model.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit!(logistic_model, IpoptSolver(print_level=5))
@show logistic_model.β
@show logistic_model.Σ
@show loglikelihood!(logistic_model, true, false)
@show logistic_model.∇β
@show logistic_model.∇Σ


##
using MixedModels, RData
datf = joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")
const dat = Dict(Symbol(k) => v for (k, v) in load(datf));
data = dat[:VerbAgg]
using DataFrames

out = map(x -> strip(String(x)) == "N" ? 0.0 : 1.0, data[!, :r2])
d = Bernoulli()
D = typeof(d)
gidx1 = data[!, :item] .== "S1WantCurse"
gidx2 = data[!, :item] .== "S1WantScold"
n1 = count(gidx1)
n2 = count(gidx2)
y1 = Float64.(out[gidx1, 1])
y2 = Float64.(out[gidx2, 1])
intercept1 = ones(n1, 1)
intercept2 = ones(n2, 1)
Anger1 = Float64.(data[gidx1, :a])
Anger2 = Float64.(data[gidx2, :a])
X1 = ones(n1, 1)# Anger1]
X2 = ones(n2, 1) ## Anger2]
V1 = [ones(n1, n1)]
V2 = [ones(n2, n2)]
gcs2 = Vector{GLMCopulaVCObs{Float64, D}}(undef, 2)
gcs2[1] = GLMCopulaVCObs(y1, X1, V1, d)
gcs2[2] = GLMCopulaVCObs(y2, X2, V2, d)
logistic_model_MM = GLMCopulaVCModel(gcs2);
initialize_model!(logistic_model_MM); # this will also standardize the residuals
fill!(logistic_model_MM.Σ, 1)
update_Σ!(logistic_model_MM)
@show logistic_model_MM.β
@show logistic_model_MM.Σ
copulogl = loglikelihood!(logistic_model_MM, true, false) # -403.6009612261338
@show logistic_model_MM.∇β
@show logistic_model_MM.∇Σ
# fit model using NLP on profiled loglikelihood
@info "MLE:"

# Ipopt does not like this problem.
#GLMCopula.fit!(logistic_model_MM, IpoptSolver(print_level=5))

# gives closest estimate
@time GLMCopula.fit!(logistic_model_MM, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
#@time GLMCopula.fit!(logistic_model_MM, NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000))
#@time GLMCopula.fit!(logistic_model_MM, NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000))

@show logistic_model_MM.β
@show logistic_model_MM.Σ
@show loglikelihood!(logistic_model_MM, true, false)
@show logistic_model_MM.∇β
@show logistic_model_MM.∇Σ

# using mixedmodels
y_full = vcat(y1, y2)
X_full = vcat(X1, X2)
group1 = data[gidx1, :item]
group2 = data[gidx2, :item]
group = vcat(group1, group2)
Df = DataFrame(y = y_full, g = group)
verbaggform = @formula(y ~ 1 + (1|g));

gm2 = fit(GeneralizedLinearMixedModel, verbaggform, Df, Bernoulli())

GLMMlogl = loglikelihood(gm2)

@test copulogl >= GLMMlogl

# j =  20 test case
# ∇mu = (exp(gc.η[j])/(1 + exp(gc.η[j]))^2) .* transpose(gc.X[j, :])
# 1×2 Array{Float64,2}:
#  0.0145864  0.0802249
#
# julia> numerator = (exp(gc.η[j])*(1-exp(gc.η[j])))
# -4361.322990346057
#
# julia> denom = (1+exp(gc.η[j]))^3
# 308124.14082174253
#
# julia> numerator/denom .* transpose(gc.X[j, :])
# 1×2 Array{Float64,2}:
#  -0.0141544  -0.0778494
#
# julia> ∇σβ = numerator/denom .* transpose(gc.X[j, :])
# 1×2 Array{Float64,2}:
#  -0.0141544  -0.0778494
#
# check res∇β
# julia> -inv(sd20).*∇mu - inv(2*gc.varμ[j])*gc.res[j] .* ∇σβ
# 1×2 Array{Float64,2}:
#  -0.0612945  -0.33712
#
# julia> gc.∇resβ
# 20×2 Array{Float64,2}:
#  -0.0948136  -0.0474068
#  -0.114433   -0.0858251
#  -0.138113   -0.138113
#  -0.166693   -0.208366
#  -0.201187   -0.301781
#  -0.242819   -0.424933
#  -1.02957    -1.80175
#  -0.293066   -0.586131
#  -0.706794   -1.59029
#  -0.426904   -1.06726
#  -0.485208   -1.33432
#  -0.621863   -1.86559
#  -0.333091   -1.08255
#  -0.905856   -3.1705
#  -0.189459   -0.757837
#  -0.156976   -0.667148
#  -0.130062   -0.58528
#  -0.107763   -0.511873
#  -0.0892866  -0.446433
#  -0.0612945  -0.33712
#
