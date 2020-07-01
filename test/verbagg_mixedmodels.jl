println()
@info "Testing Logistic Regression on VerbAgg dataset in  MixedModels.jl"

using MixedModels, DataFrames
using GLM, RData, GLMCopula, Test

datf = joinpath(dirname(pathof(MixedModels)),"..","test","dat.rda")
const dat = Dict(Symbol(k)=>v for (k,v) in load(datf));
data = dat[:VerbAgg]

groups = unique(data[!, :id])
out = map(x -> strip(String(x)) == "N" ? 0.0 : 1.0, data[!, :r2])
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
# gcm_all.β .= [-1.0314; 0.0458053]
# update σ2 and τ from β using the MM algorithm
update_Σ!(gcm_all)

gc = gcm_all.data[1]
β = gcm_all.β
τ = gcm_all.τ[1]
Σ = gcm_all.Σ

# julia> glm_gradient(gcm_all)
# 1.4694485628297116e-9
# 6.816210174065418e-8

# julia> copula_gradient(gcm_all)
# 2-element Array{Float64,1}:
#   39.84849398496692
#  853.8308241932496

# julia> inform
# 2×2 Array{Float64,2}:
#   1878.33   37599.2
#  37599.2   796020.0
#
# julia> score
# 2-element Array{Float64,1}:
#  1.4574270124079192e-9
#  6.775510374978921e-8

@test loglikelihood!(gcm_all, true, false) ==  -4967.482483496547
#@time GLMCopula.fit!(gcm_all, IpoptSolver(print_level=5))
@time fit2!(gcm_all, IpoptSolver(print_level=5))
@time GLMCopula.fit!(gcm_all, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
@show gcm_all.β
@show gcm_all.Σ
@show loglikelihood!(gcm_all, true, false)
@show gcm_all.∇β
@show gcm_all.∇Σ
#
Df = DataFrame(y = out, int = ones(length(out)), a = data[:, :a], id = data[:, :id])
verbaggform = @formula(y ~ 1 + a + (1|id));
gm_all = fit(GeneralizedLinearMixedModel, verbaggform, Df, Bernoulli())
glmm_logl = loglikelihood(gm_all)

#
# # # @time GLMCopula.fit!(gcm2, NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000))
# # # Ok now what if I just have the first 2 groups?
# out = map(x -> strip(String(x)) == "N" ? 0.0 : 1.0, data[!, :r2])
# d = Bernoulli()
# D = typeof(d)
# gcs2 = Vector{GLMCopulaVCObs{Float64, D}}(undef, 2)
# gidx1 = data[!, :id] .== "1"
# gidx2 = data[!, :id] .== "2"
# gidx3 = data[!, :id] .== "3"
# gidx4 = data[!, :id] .== "4"
# gidx5 = data[!, :id] .== "5"
# n1 = count(gidx1)
# n2 = count(gidx2)
# y1 = Float64.(out[gidx1, 1])
# y2 = Float64.(out[gidx2, 1])
# intercept1 = ones(n1, 1)
# intercept2 = ones(n2, 1)
# Anger1 = Float64.(data[gidx1, :a])
# Anger2 = Float64.(data[gidx2, :a])
# X1 = [intercept1 Anger1]
# X2 = [intercept2 Anger2]
# V1 = [ones(n1, n1)]
# V2 = [ones(n2, n2)]
# gcs2[1] = GLMCopulaVCObs(y1, X1, V1, d)
# gcs2[2] = GLMCopulaVCObs(y2, X2, V2, d)
# gcm2 = GLMCopulaVCModel(gcs2);
# initialize_model!(gcm2); # this will also standardize the residuals
# #update_res!(gcm2)
# fill!(gcm2.Σ, 1.0)
# update_Σ!(gcm2)
# loglikelihood!(gcm2, true, false)
# # @time GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, derivative_test = "first-order"))
# #GLMCopula.fit!(gcm2, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
#
# y_full = vcat(gcm_all.data[1].y, gcm_all.data[2].y, gcm_all.data[3].y, gcm_all.data[4].y, gcm_all.data[5].y)
# X_full = vcat(gcm_all.data[1].X, gcm_all.data[2].X, gcm_all.data[3].X, gcm_all.data[4].X, gcm_all.data[5].X)
# group1 = data[gidx1, :id]
# group2 = data[gidx2, :id]
# group3 = data[gidx3, :id]
# group4 = data[gidx4, :id]
# group5 = data[gidx5, :id]
# group = vcat(group1, group2, group3, group4, group5)
# Df2 = DataFrame(y = y_full, int = X_full[:, 1], a = X_full[:, 2], id = group)
# verbaggform = @formula(y ~ 1 + a + (1|id));
#
# gm5 = fit(GeneralizedLinearMixedModel, verbaggform, Df2, Bernoulli())
# logl_gm5 = loglikelihood(gm5)

#
# fill!(gcm2.Σ, 1)
# # update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
# update_Σ!(gcm2)
# @show gcm2.τ
# @show gcm2.Σ;
# loglikelihood!(gcm2, true, false)
#
# GLMCopula.fit!(gcm2, NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000))
# #GLMCopula.fit!(gcm2, NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000))
#
# copulalogl2 = loglikelihood!(gcm2, true, false)
# @test mixedmodelslogl2 <= copulalogl2
# the gradient with respect to beta seems off
