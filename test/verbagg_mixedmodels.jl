println()
@info "Testing Logistic Regression on VerbAgg dataset in  MixedModels.jl"

using MixedModels, DataFrames
using GLM, RData, GLMCopula

datf = joinpath(dirname(pathof(MixedModels)),"..","test","dat.rda")
const dat = Dict(Symbol(k)=>v for (k,v) in load(datf));
data = dat[:VerbAgg]

groups = unique(data[!, :item])[1:5]
out = map(x -> strip(x) == "N" ? 0.0 : 1.0, data[!, :r2])
n = length(groups)
d = Bernoulli()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
 for (i, grp) in enumerate(groups)
    gidx = data[!, :item] .== grp
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
# update σ2 and τ from β using the MM algorithm
fill!(gcm_all.Σ, 1)
update_Σ!(gcm_all)
loglikelihood!(gcm_all, true, true) # -5153.9040599608525
# @time GLMCopula.fit!(gcm_all, IpoptSolver(print_level=5))
@time GLMCopula.fit!(gcm_all, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
@show gcm_all.β
@show gcm_all.Σ
@show loglikelihood!(gcm_all, true, false)
@show gcm_all.∇β
@show gcm_all.∇Σ

out = map(x -> strip(x) == "N" ? 0.0 : 1.0, data[!, :r2])
n = length(groups)
d = Bernoulli()
D = typeof(d)
gcs2 = Vector{GLMCopulaVCObs{Float64, D}}(undef, 2)
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
X1 = [intercept1 Anger1]
X2 = [intercept2 Anger2]
V1 = [ones(n1, n1)]
V2 = [ones(n2, n2)]
gcs2[1] = GLMCopulaVCObs(y1, X1, V1, d)
gcs2[2] = GLMCopulaVCObs(y2, X2, V2, d)
gcm2 = GLMCopulaVCModel(gcs2);
initialize_model!(gcm2); # this will also standardize the residuals

y_full = vcat(y1, y2)
X_full = vcat(X1, X2)
group1 = data[gidx1, :item]
group2 = data[gidx2, :item]
group = vcat(group1, group2)
Df = DataFrame(y = y_full, int = X_full[:, 1], a = X_full[:, 2], g = group)
verbaggform = @formula(y ~ 1 + a + (1|g));

gm2 = fit(GeneralizedLinearMixedModel, verbaggform, Df, Bernoulli())
mixedmodelslogl = loglikelihood(gm2)

fill!(gcm2.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm2)
@show gcm2.τ
@show gcm2.Σ;
loglikelihood!(gcm2, true, false)

#GLMCopula.fit!(gcm, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
GLMCopula.fit!(gcm, NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000))
GLMCopula.fit!(gcm, NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000))

copulalogl = loglikelihood!(gcm, true, false)
@test mixedmodelslogl <= copulalogl
# the gradient with respect to beta seems off
