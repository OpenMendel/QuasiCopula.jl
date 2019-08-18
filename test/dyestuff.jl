module DyestuffTest

println()
@info "Dyestuff example"

using GLMCopula, RDatasets, Test

# Dataframe with columns: Batch (Categorical), Yield (Int32)
dyestuff = dataset("lme4", "Dyestuff")
groups = unique(dyestuff[!, :Batch])
n, p, m = length(groups), 1, 1
gcs = Vector{GaussianCopulaVCObs{Float64}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dyestuff[!, :Batch] .== grp
    ni = count(gidx)
    y = Float64.(dyestuff[gidx, :Yield])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V)
end
gcm = GaussianCopulaVCModel(gcs);

# initialize β and τ from least square solution
init_β!(gcm)
@show gcm.β
@show gcm.τ
# update σ2 from β and τ using the MM algorithm
standardize_res!(gcm)
update_quadform!(gcm, true)
fill!(gcm.Σ, 1)
update_Σ!(gcm)
@show gcm.Σ;
@test loglikelihood!(gcm, true, false) ≈ -165.27260315
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

# fit model using NLP on profile loglikelihood
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@test loglikelihood!(gcm, true, false) ≈  -163.40909306
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end