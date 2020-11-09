println()
@info "Testing Logistic Regression on VerbAgg dataset in  MixedModels.jl"

using MixedModels, DataFrames
using GLM, RData, GLMCopula, Test

datf = joinpath(dirname(pathof(MixedModels)),"..", "test", "dat.rda")
const dat = Dict(Symbol(k)=>v for (k,v) in load(datf));
data = dat[:VerbAgg]

#transform the yes no outcome to 1 0 for logistic regression
out = map(x -> strip(String(x)) == "N" ? 0.0 : 1.0, data[!, :r2])

# using the mixed models package we get a higher loglikelihood
Df = DataFrame(y = out, int = ones(length(out)), a = data[:, :a], id = data[:, :id])
verbaggform = @formula(y ~ 1 + a + (1|id));
gm_all = fit(GeneralizedLinearMixedModel, verbaggform, Df, Bernoulli())
@show loglikelihood(gm_all) # -4749.666193096799

groups = unique(data[!, :id])
n = length(groups)
d = Bernoulli()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
 for (i, grp) in enumerate(groups)
    gidx = data[!, :id] .== string(grp)
    ni = count(gidx)
    y = Float64.(out[gidx, 1])
    intercept = ones(ni, 1)
    Anger = Float64.(data[gidx, :a])
    X = [intercept Anger]
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm_all = GLMCopulaVCModel(gcs);

initialize_model!(gcm_all)
fill!(gcm_all.Σ, 1.0)
gcm_all.β .= [-1.0314; 0.0458053] # from the mixed models package
# update σ2 from β using the MM algorithm
update_Σ!(gcm_all)

@test copula_loglikelihood(gcm_all) ==  -4967.48248349655
@test loglikelihood!(gcm_all, true, true)  == -4967.48248349655
@time fit2!(gcm_all, IpoptSolver(print_level = 5, max_iter = 100, derivative_test = "first-order", hessian_approximation = "limited-memory"))

gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β
fill!(gcm.Σ, 1.0)
update_Σ!(gcm)
@show GLMCopula.loglikelihood!(gcm, true, true)
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))

#@time fit2!(gcm_all, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
@show gcm_all.β
@show gcm_all.Σ
@show gcm_all.∇β
#
