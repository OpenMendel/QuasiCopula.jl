module Poisson_Logistic_Test

println()
@info "Testing Logistic and Poisson using GLM.jl and solvers"

using Statistics, Distributions, LinearAlgebra, GLM, Test, RDatasets

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, GLMCopula
using LinearAlgebra: BlasReal, copytri!
using Ipopt
using NLopt

# Ken's test data
function create_gcm_poisson(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{glm_VCobs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
             X = reshape(X, 14, 1);
             y = [0., 1 ,2 ,3, 1, 4, 9, 18, 23, 31, 20, 25, 37, 45];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = glm_VCobs(y, X, V, dist)
           end
           gcms = glm_VCModel(Vector(gcs))
           return gcms
       end

gcm = create_gcm_poisson(2, Poisson());

copyto!(gcm.β, [0.28503887444394366])

@show gcm.β
# update σ2 from β using the MM algorithm
fill!(gcm.Σ, 1)
update_Σ!(gcm)
@show gcm.Σ;
@test loglikelihood!(gcm, true, false) ≈ 471.19671943091146*2
@show gcm.∇β
@show gcm.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@show gcm.β
@show gcm.Σ
@show loglikelihood!(gcm, true, false)
@show gcm.∇β
@show gcm.∇Σ

# # Ken's test data
# # I want to form the model using the test data Ken has, each obs object will be the same.
function create_gcm_logistic(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{glm_VCobs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = glm_VCobs(y, X, V, dist)
           end
           gcms = glm_VCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());
logistic_β = [ -4.077713431087562
                 1.5046454283733053]
copyto!(logistic_model.β, logistic_β)
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
end
