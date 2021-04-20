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
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dyestuff[!, :Batch] .== grp
    ni = count(gidx)
    y = Float64.(dyestuff[gidx, :Yield])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
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

# gc = gcm.data[1]
@test copula_gradient(gcm) ≈ [0.06561148856048]
# @test copula_hessian(gcm) ≈ [-0.01964435497982633]

@show loglikelihood!(gcm, true, true)
# fit model using NLP on profiled loglikelihood
@info "MLE:"
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))

@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))

@show gcm.β
@show gcm.τ
@show gcm.Σ
@test copula_loglikelihood(gcm)[1] ≈ -163.35545423301846
@show loglikelihood!(gcm, true, true)
@test loglikelihood!(gcm, true, false) ≈ -163.35545423301846
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end
