using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
using Test, BenchmarkTools, LoopVectorization
import StatsBase: sem

p_fixed = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p_fixed)
rtrue = 10.0
σ2true = [0.5]
ρtrue = [0.5]

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
samplesize = 10000

st = time()
currentind = 1
d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 25 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    # X = [ones(ni) randn(ni, p - 1)]
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    η = X * βtrue
    μ = exp.(η)
    p = rtrue ./ (μ .+ rtrue)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # push!(Ystack, y)
    V = [ones(ni, ni)]
    # V = [ones(ni, ni)]
    gcs[i] = NBCopulaARObs(y, X, d, link)
end

# form model
gcm = NBCopulaARModel(gcs);

initialize_model!(gcm)
gc = gcm.data[1];
gc2 = deepcopy(gc);
gc3 = deepcopy(gc);
ρ = gcm.ρ[1]

function std_res_differential_old!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}
    ) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @inbounds for j in 1:gc.n
        gc.∇μβ[j, :] .= gc.dμ[j] .* @view(gc.X[j, :])
        gc.∇σ2β[j, :] .= (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) .* @view(gc.∇μβ[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) .* @view(gc.∇μβ[j, :]) .- (0.5 * inv(gc.varμ[j])) .* gc.res[j] .* @view(gc.∇σ2β[j, :])
    end
    nothing
end

function std_res_differential_new!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}
    ) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @inbounds for i in 1:gc.p
        @inbounds for j in 1:gc.n
            gc.∇μβ[j, i] = gc.dμ[j] .* gc.X[j, i]
            gc.∇σ2β[j, i] = (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) .* gc.∇μβ[j, i]
            gc.∇resβ[j, i] = -inv(sqrt(gc.varμ[j])) .* gc.∇μβ[j, i] .- (0.5 * inv(gc.varμ[j])) .* gc.res[j] .* gc.∇σ2β[j, i]
        end
    end
    nothing
end

function std_res_differential_new3!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}
    ) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @turbo for i in 1:gc.p
         for j in 1:gc.n
            gc.∇resβ[j, i] = -inv(sqrt(gc.varμ[j])) * gc.dμ[j] * gc.X[j, i] - (0.5 * inv(gc.varμ[j])) * gc.res[j] * (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) *  gc.dμ[j] * gc.X[j, i]
        end
    end
    nothing
end

std_res_differential_old!(gc)
a = gc.∇resβ
std_res_differential_new!(gc2)
b = gc2.∇resβ
std_res_differential_new3!(gc3)
c = gc3.∇resβ
@test a ≈ b
@test a ≈ c
@benchmark std_res_differential_old!($gc)
@benchmark std_res_differential_new!($gc)
@benchmark std_res_differential_new3!($gc)
