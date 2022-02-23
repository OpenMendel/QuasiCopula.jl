using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices, Test, BenchmarkTools
using LoopVectorization
using LinearAlgebra: BlasReal, copytri!
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
# βtrue = [log(5.0)]
σ2true = [0.5]
ρtrue = [0.5]
trueparams = [βtrue; ρtrue; σ2true]

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = vec[i - 1] * ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end

#simulation parameters
samplesize = 10_000

st = time()
currentind = 1
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 25# number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(1234)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Poisson(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

# form model
gcm = GLMCopulaARModel(gcs);

initialize_model!(gcm)
gc = gcm.data[1];
ρ = gcm.ρ[1]

function update_res_old!(
   gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaCSObs, GLMCopulaARObs, NBCopulaARObs, NBCopulaCSObs},
   β::Vector)
   mul!(gc.η, gc.X, β)
   @inbounds for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(gc.link, gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i]) # Note: for negative binomial, d.r is used
       gc.dμ[i] = GLM.mueta(gc.link, gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.w1[i] * gc.dμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end

function update_res_new!(
   gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaCSObs, GLMCopulaARObs, NBCopulaARObs, NBCopulaCSObs},
   β::Vector)
   mul!(gc.η, gc.X, β)
   @turbo for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(gc.link, gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i]) # Note: for negative binomial, d.r is used
       gc.dμ[i] = GLM.mueta(gc.link, gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.w1[i] * gc.dμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end
a = update_res_old!(gc, gcm.β)
b = update_res_new!(gc, gcm.β)
@test a ≈ b
@benchmark update_res_old!($gc, $gcm.β)
@benchmark update_res!($gc, $gcm.β)

gc2 = deepcopy(gc);

function standardize_res_old!(
    gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaARObs, GLMCopulaCSObs, NBCopulaARObs, NBCopulaCSObs, Poisson_Bernoulli_VCObs}
    )
    @inbounds for j in eachindex(gc.y)
        gc.res[j] /= sqrt(gc.varμ[j])
    end
    gc.res
end

function standardize_res_new!(
    gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaARObs, GLMCopulaCSObs, NBCopulaARObs, NBCopulaCSObs, Poisson_Bernoulli_VCObs}
    )
    @turbo for j in eachindex(gc.y)
        gc.res[j] /= sqrt(gc.varμ[j])
    end
    gc.res
end
a = standardize_res_old!(gc)
b = standardize_res_new!(gc2)
@test a ≈ b
@benchmark standardize_res_old!($gc)
@benchmark standardize_res_new!($gc)

function std_res_differential_old!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
    @inbounds for i in 1:size(gc.X, 2)
        @inbounds for j in 1:length(gc.y)
            gc.∇resβ[j, i] = gc.X[j, i]
            gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
        end
    end
    gc.∇resβ
end

function std_res_differential_new!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
    @turbo for i in 1:size(gc.X, 2)
         for j in 1:length(gc.y)
            gc.∇resβ[j, i] = gc.X[j, i]
            gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
        end
    end
    gc.∇resβ
end

a = std_res_differential_old!(gc)
b = std_res_differential_new!(gc)
@test a ≈ b
@benchmark std_res_differential_old!($gc)
@benchmark std_res_differential_new!($gc)
