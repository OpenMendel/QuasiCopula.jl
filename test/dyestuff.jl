module DyestuffTest

println()
@info "Dyestuff example"

using GLMCopula, RDatasets, Test, GLM

# Dataframe with columns: Batch (Categorical), Yield (Int32)
dyestuff = dataset("lme4", "Dyestuff")
groups = unique(dyestuff[!, :Batch])
n, p, m = length(groups), 1, 1
d = Normal()
D = typeof(d)
gcs = Vector{GaussianCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dyestuff[!, :Batch] .== grp
    ni = count(gidx)
    y = Float64.(dyestuff[gidx, :Yield])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V, d)
end
gcm = GaussianCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
init_β!(gcm)
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;
@test loglikelihood!(gcm, true, false) ≈ -164.00082379
@show gcm.∇β
# @show gcm.∇τ
@show gcm.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@test loglikelihood!(gcm, true, false) ≈ -163.35545251
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end

### linear mixed model with dyestuff not estimating sigma now
# keep debugging here
# Dataframe with columns: Batch (Categorical), Yield (Int32)
# dyestuff = dataset("lme4", "Dyestuff")
# groups = unique(dyestuff[!, :Batch])
# n, p, m = length(groups), 1, 1
# d = Normal()
# D = typeof(d)
# gcs = Vector{glm_VCobs{Float64, D}}(undef, n)
# for (i, grp) in enumerate(groups)
#     gidx = dyestuff[!, :Batch] .== grp
#     ni = count(gidx)
#     y = Float64.(dyestuff[gidx, :Yield])
#     X = ones(ni, 1)
#     V = [ones(ni, ni)]
#     gcs[i] = glm_VCobs(y, X, V, d)
# end
# gcm_normal = glm_VCModel(gcs);
#
# function init_β!(
#     gcm::glm_VCModel{T, D}
#     ) where {T <: BlasReal, D}
#     # accumulate sufficient statistics X'y
#     xty = zeros(T, gcm.p)
#     for i in eachindex(gcm.data)
#         BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
#     end
#     # least square solution for β s.t gcm.β = inv(cholesky(Symmetric(gcm.XtX)))*xty
#     ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
#     gcm.β
# end
# # initialize β and τ from least square solution
# @info "Initial point:"
# init_β!(gcm_normal)
# @show gcm_normal.β
# # update σ2 and τ from β using the MM algorithm
# fill!(gcm_normal.Σ, 1)
# # update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
# # update_Σ!(gcm_normal)
#
#
# @show gcm_normal.Σ;
# @show loglikelihood!(gcm_normal, true, false) #≈ -164.00082379
# @show gcm_normal.∇β
# # @show gcm.∇τ
# @show gcm_normal.∇Σ
#
# # fit model using NLP on profiled loglikelihood
# @info "MLE:"
# @time GLMCopula.fit!(gcm_normal, IpoptSolver(print_level=5))
# @show gcm_normal.β
# @show gcm_normal.Σ
# @show loglikelihood!(gcm_normal, true, false)#≈ -163.35545251
# @show gcm_normal.∇β
# @show gcm_normal.∇Σ
