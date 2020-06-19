println()
@info "Testing Poisson on Ken's test data using glm_regress as starting point, then using GLM.jl for loglikelihood and solvers for updating beta"

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Statistics, GLMCopula, Test
using LinearAlgebra: BlasReal, copytri!
# @reexport using Ipopt
# @reexport using NLopt

# Ken's test data
function create_gcm_poisson(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
             X = reshape(X, 14, 1);
             y = [0., 1 ,2 ,3, 1, 4, 9, 18, 23, 31, 20, 25, 37, 45];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

gcm = create_gcm_poisson(2, Poisson());

initialize_model!(gcm);
@test gcm.β == [0.28503887444394366]
# update σ2 from β using the MM algorithm
fill!(gcm.Σ, 1)
update_Σ!(gcm)

@test gcm.Σ == [1.2623029177859688e-7]
#@show gcm.Σ;
@test loglikelihood!(gcm, true, false) ≈ 471.19671943091146*2  # 942.3934374279268
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
#-102.95885585666409 + 8.646784118270125 = -94.31207173839397 for obs 14
