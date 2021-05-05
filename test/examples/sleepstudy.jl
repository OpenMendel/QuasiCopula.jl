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
    Zi = Xi
    gcs[i] = GaussianCopulaLMMObs(yi, Xi, Zi)
end
gcm = GaussianCopulaLMMModel(gcs);

# initialize β and τ from least squares solution
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

# # initialize from LMM estimate
# @info "Initial point:"
# gcm.β .= [251.405, 10.4673]
# gcm.τ[1] = inv(654.94145)
# gcm.Σ .= [565.51069 10.87591; 10.87591 32.68212]
# @show loglikelihood!(gcm, true, false)
# @show gcm.∇β
# @show gcm.∇τ
# @show gcm.∇Σ

# fit model using NLP
@info "MLE:"
# @time GLMCopula.fit!(gcm, NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000))
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@show loglikelihood!(gcm, true, false)
# @test loglikelihood!(gcm, true, false) ≈ -163.35545251
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end