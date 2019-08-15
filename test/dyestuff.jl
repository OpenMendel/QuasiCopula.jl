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
fill!(gcm.σ2, 1)
update_σ2!(gcm)
@show gcm.σ2;
@test loglikelihood!(gcm, true, false) ≈ -165.27260315
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇σ2

# fit model using NLP on profile loglikelihood
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=0))
@show gcm.β
@show gcm.τ
@show gcm.σ2
@test loglikelihood!(gcm, true, false) ≈  -163.40909306
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇σ2

end