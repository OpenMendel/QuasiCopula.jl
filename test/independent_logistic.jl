println()
@info "Testing Logistic Regression: the independent case confirms independent"

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Statistics, GLMCopula, Test
using LinearAlgebra: BlasReal, copytri!

# # Ken's test data
# # I want to form the model using the test data Ken has, each obs object will be the same.
function create_gcm_logistic(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());
logistic_β = [ -4.077713431087562
                 1.5046454283733053]

initialize_model!(logistic_model)
# @test logistic_model.β
# update σ2 from β using the MM algorithm
fill!(logistic_model.Σ, 1)
update_Σ!(logistic_model)

#copyto!(logistic_model.β, logistic_β)
@show loglikelihood!(logistic_model, true, false) #≈  -8.029878464344675*2
@show logistic_model.∇β
@show logistic_model.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit!(logistic_model, IpoptSolver(print_level=5))
@show logistic_model.β
@show logistic_model.Σ
@show loglikelihood!(logistic_model, true, false)
@show logistic_model.∇β
@show logistic_model.∇Σ
#
# # use mixedmodels to check if we get similar solutions for independent case
# y_full = vcat(logistic_model.data[1].y, logistic_model.data[2].y)
# X_full = vcat(logistic_model.data[1].X, logistic_model.data[2].X)
# group1 = string.(zeros(20))
# group2 = string.(ones(20))
# group = vcat(group1, group2)
# Df = DataFrame(y = y_full, int = X_full[:, 1], a = X_full[:, 2], g = group)
# verbaggform = @formula(y ~ 1 + a + (1|g));
#
# gm2 = fit(GeneralizedLinearMixedModel, verbaggform, Df, Bernoulli())
# mixedmodelslogl2 = loglikelihood(gm2)
#
#
# # j =  20 test case
# # ∇mu = (exp(gc.η[j])/(1 + exp(gc.η[j]))^2) .* transpose(gc.X[j, :])
# # 1×2 Array{Float64,2}:
# #  0.0145864  0.0802249
# #
# # julia> numerator = (exp(gc.η[j])*(1-exp(gc.η[j])))
# # -4361.322990346057
# #
# # julia> denom = (1+exp(gc.η[j]))^3
# # 308124.14082174253
# #
# # julia> numerator/denom .* transpose(gc.X[j, :])
# # 1×2 Array{Float64,2}:
# #  -0.0141544  -0.0778494
# #
# # julia> ∇σβ = numerator/denom .* transpose(gc.X[j, :])
# # 1×2 Array{Float64,2}:
# #  -0.0141544  -0.0778494
# #
# # check res∇β
# # julia> -inv(sd20).*∇mu - inv(2*gc.varμ[j])*gc.res[j] .* ∇σβ
# # 1×2 Array{Float64,2}:
# #  -0.0612945  -0.33712
# #
# # julia> gc.∇resβ
# # 20×2 Array{Float64,2}:
# #  -0.0948136  -0.0474068
# #  -0.114433   -0.0858251
# #  -0.138113   -0.138113
# #  -0.166693   -0.208366
# #  -0.201187   -0.301781
# #  -0.242819   -0.424933
# #  -1.02957    -1.80175
# #  -0.293066   -0.586131
# #  -0.706794   -1.59029
# #  -0.426904   -1.06726
# #  -0.485208   -1.33432
# #  -0.621863   -1.86559
# #  -0.333091   -1.08255
# #  -0.905856   -3.1705
# #  -0.189459   -0.757837
# #  -0.156976   -0.667148
# #  -0.130062   -0.58528
# #  -0.107763   -0.511873
# #  -0.0892866  -0.446433
# #  -0.0612945  -0.33712
# #
