module DyestuffTest

println()
@info "Dyestuff example"

using RDatasets, Test, GLM, LinearAlgebra, GLMCopula
using LinearAlgebra: BlasReal, copytri!
# Dataframe with columns: Batch (Categorical), Yield (Int32)
dyestuff = dataset("lme4", "Dyestuff")
groups = unique(dyestuff[!, :Batch])
n, p, m = length(groups), 1, 1
d = Normal()
link = IdentityLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dyestuff[!, :Batch] .== grp
    ni = count(gidx)
    y = Float64.(dyestuff[gidx, :Yield])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
initialize_model!(gcm);
#gcm.β .= [1527.4999999999998]
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;

@show loglikelihood!(gcm, true, true)
# gc = gcm.data[1]
@test copula_gradient(gcm) ≈ [0.06561148856048]
@test gcm.data[1].∇β ≈ [0.01997344639809115]

# # without extra hessian term
# @test gcm.Hβ ≈ [-0.019644354979826334]
# @test gcm.data[1].Hβ ≈ [-14.894316220056414]

# with the extra hessian term
@test gcm.Hβ ≈ [-0.0060445770824270485]
@test gcm.data[1].Hβ ≈ [-7.217632930773922]

# fit model using NLP on profiled loglikelihood
@info "MLE:"
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
# 11 iterations 1 second
@show gcm.β
@show gcm.τ
@show gcm.Σ
# @test copula_loglikelihood(gcm)[1] ≈ -163.35545423301846
@show loglikelihood!(gcm, true, true)
@test loglikelihood!(gcm, true, true) ≈ -163.35545423301846
@show gcm.∇β
@show gcm.Hβ

end
