@reexport using Distributions
import Distributions: rand
export MultivariateMix
# ## create new type not in Distributions for multivariate mixture

"""
    MultivariateMix(vecd::AbstractVector{<:UnivariateDistribution}, Γ)
An N dimensional `MultivariateMix` constructed from a vector of N `UnivariateDistribution`s, and the user-specified covariance matrix, Γ.
Since Distributions.jl does not currently allow for multivariate vectors with mixed continuous and discrete components, this type is not of super type Distributions.MultivariateDistribution.
We will pre-allocate a Vector of `ContinuousUnivariateCopula`s and `DiscreteUnivariateCopula`s to then fill in the appropriate constants c0, c1, c2, recursively. 
"""
struct MultivariateMix{
    V<:AbstractVector{UnivariateDistribution},
    T<: BlasReal
} 
    vecd::V
    Γ::Matrix{T}
    gc_obs::Vector{Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}}
    function MultivariateMix(vecd::V, Γ::Matrix{T}) where {T <: BlasReal,
        V<:AbstractVector{UnivariateDistribution}}
        n = length(vecd)
        gc_obs = Vector{Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}}(undef, n)
        return new{V, T}(vecd, Γ, gc_obs)
    end
end

# ### we need a function that is like rand, but cannot use the distributions framework.
function rand(gc_vec::MultivariateMix{V, T},
            Y::Vector{T},
            res::Vector{T}) where {T <: BlasReal, V<:AbstractVector{UnivariateDistribution}}
    for i in 1:length(gc_vec.vecd)
        # form constants for conditional density of i given 1, ..., i-1
        gc_vec.gc_obs[i] = pdf_constants(gc_vec.Γ, res, i, gc_vec.vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = rand(gc_vec.gc_obs[i])
        # update residuals 1,..., i-1
        res[i] = update_res!(Y[i], res[i], gc_vec.gc_obs[i])
     end
    res[end] = update_res!(Y[end], res[end], gc_vec.gc_obs[end])
    Y
end 
