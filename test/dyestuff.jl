module DyestuffTest

println()
@info "Dyestuff example"

using RDatasets, Test, GLM, GLMCopula, LinearAlgebra

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
initialize_model!(gcm)
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;

# making sure the loglikelihood values are the same
gc = gcm.data[1]
β = gcm.β
τ = gcm.τ[1]
Σ = gcm.Σ
needgrad = true;  needhess = true
n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
needgrad = needgrad || needhess
update_res!(gc, β)
if gc.d  ==  Normal()
    sqrtτ = sqrt(τ)
    standardize_res!(gc, sqrtτ)
else
    sqrtτ = 1.0
    standardize_res!(gc)
end

@test sqrtτ == 0.018211123993574548

@test gc.res[1] == 0.31869466988755873

if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇Σ, 0)
        fill!(gc.∇resβ, 0.0)
        GLMCopula.std_res_differential!(gc)
        fill!(gc.∇τ, 0)
    end

needhess && fill!(gc.Hβ, 0)
@show gc.res
# evaluate copula loglikelihood
tsum = dot(Σ, gc.t)
#@show tsum
logl = - log(1 + tsum)
@test logl ≈  -1.1502673245404629
@test tsum ≈ 2.159037285014138

for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        #gc.storage_n
        if needgrad # ∇β stores X'*W*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β)
            @show gc.∇β
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end

# @test gc.∇β ≈ [-8.846661533432094]
qsum  = dot(Σ, gc.q)
logl += log(1 + qsum)
@test logl ≈ -0.11620740109213168

logl += component_loglikelihood(gc, τ, 0.0)

@test logl ≈ -27.795829678091447

@test copula_loglikelihood(gcm) ≈ -164.00082379
score, hess = beta_gradient_hessian(gcm)
@test score == [0.06561148856048625]
@test hess[1] == -0.019644354979826334

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit2!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.τ
@show gcm.Σ
@test copula_loglikelihood(gcm)[1] ≈ -163.355454233
@show gcm.∇β
@show gcm.∇τ
@show gcm.∇Σ

end
