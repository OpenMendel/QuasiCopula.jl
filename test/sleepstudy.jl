module SleepstudyTest

println()
@info "Sleepstudy example"

using GLMCopula, RDatasets, Test, LinearAlgebra

# Dataframe with columns: Reaction (Float64), Days (Int32), Subject (Categorical)
# 180 rows
sleepstudy = dataset("lme4", "sleepstudy")
groups = unique(sleepstudy[!, :Subject])
n, p, m = length(groups), 2, 1
gcs = Vector{GaussianCopulaLMMObs{Float64}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = sleepstudy[!, :Subject] .== grp
    ni = count(gidx)
    yi = Float64.(sleepstudy[gidx, :Reaction])
    Ui = Float64.(sleepstudy[gidx, :Days])
    Xi = [ones(ni, 1) Ui]
    gcs[i] = GaussianCopulaLMMObs(yi, Xi, Xi)
end
gcm = GaussianCopulaLMMModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
init_β!(gcm)
@show gcm.β
@show gcm.τ[1]
gcm.Σ .= diagm(ones(2))
@show gcm.Σ
@show loglikelihood!(gcm, true, false)
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

# fit model using NLP
@info "MLE:"
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=3))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@show loglikelihood!(gcm, true, false)
# @test loglikelihood!(gcm, true, false) ≈ -163.35545251
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end