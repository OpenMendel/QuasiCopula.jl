module DyestuffTest2

println()
@info "Dyestuff example"


using GLMCopula, RDatasets, Test, GLM, GLMCopula

# Dataframe with columns: Batch (Categorical), Yield (Int32)
dyestuff2 = dataset("lme4", "Dyestuff2")
groups = unique(dyestuff2[!, :Batch])
n, p, m = length(groups), 1, 1
d = Normal()
D = typeof(d)
gcs = Vector{GaussianCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dyestuff2[!, :Batch] .== grp
    ni = count(gidx)
    y = Float64.(dyestuff2[gidx, :Yield])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V, d)
end
gcm = GaussianCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
initialize_model!(gcm)
@show gcm.β
# update σ2 from β and τ using the MM algorithm
fill!(gcm.Σ, 1)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;
@test loglikelihood!(gcm, true, false) ≈ -81.43652408245762
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

# fit model using NLP on profile loglikelihood
@info "MLE:"
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=0))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@show loglikelihood!(gcm, true, false)# ≈ -81.43651913862902
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end
