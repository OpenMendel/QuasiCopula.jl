"""
    ∇²σ²_j(d::Distribution, l::Link, Xi::Matrix, β::Vector, j)

Computes the Hessian of the σ^2 function with respect to β for sample i (Xi) at time j
"""
function ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector, Number}, μj::Number, ηj::Number)
    c = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    return c * xj * xj' # p × p or 1 × 1
end
function ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector, Number}, zj::Number, μj::Number, ηj::Number)
    c = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    return c * zj * xj # p × 1
end

"""
    ∇²μ_j(l::Link, Xi::Matrix, β::Vector, j)

Computes the Hessian of the mean function with respect to β for sample i (Xi) at time j
"""
function ∇²μ_j(l::Link, ηj::Number, xj::Union{AbstractVector, Number})
    d²μdη² = mueta2(l, ηj)
    return d²μdη² * xj * xj' # p × p or 1 × 1
end
function ∇²μ_j(l::Link, ηj::Number, xj::Union{AbstractVector, Number}, zj::Number)
    d²μdη² = mueta2(l, ηj)
    return d²μdη² * zj * xj # p × 1
end

"""
    sigmaμ2(D::Distribution, μ::Real)

Computes d²σ²/dμ²
"""
function sigmamu2 end
sigmamu2(::Normal, μ::Real) = zero(μ)
sigmamu2(::Bernoulli, μ::Real) = -2
sigmamu2(::Poisson, μ::Real) = zero(μ)

"""
    sigmaeta(D::Distribution, μ::Real)

Computes dσ²/dμ
"""
function sigmamu end
sigmamu(::Normal, μ::Real) = zero(μ)
sigmamu(::Bernoulli, μ::Real) = one(μ) - 2μ
sigmamu(::Poisson, μ::Real) = one(μ)

"""
    mueta2(l::Link, η::Real)

Second derivative of the inverse link function `d^2μ/dη^2`, for link `L` at linear predictor value `η`.
I.e. derivative of the mueta function in GLM.jl
"""
function mueta2 end
mueta2(::IdentityLink, η::Real) = zero(η)
function mueta2(::LogitLink, η::Real)
    expabs = exp(-abs(η))
    denom = 1 + expabs
    return -expabs / denom^2 + 2expabs^2 / denom^3
end
mueta2(::LogLink, η::Real) = exp(η)

"""
    ∇²resβ_ij(dist::Distribution, link::Link, i, j)

Computes ∇²resβ_ij, the Hessian of the standardized residuals for sample i 
at the j'th measurement. 
"""
function ∇²resβ_ij(dist, link, xj, η_j, μ_j, varμ_j, res_j)
    invσ_j = inv(sqrt(varμ_j))
    ∇μ_ij  = GLM.mueta(link, η_j) * xj
    ∇σ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * xj

    # assemble 5 terms
    term1 = -invσ_j * ∇²μ_j(link, η_j, xj)
    term2 = 0.5invσ_j^3 * ∇σ²_ij * ∇μ_ij'
    term3 = -0.5 * res_j * inv(varμ_j) * ∇²σ²_j(dist, link, xj, μ_j, η_j)
    term4 = 0.5invσ_j^3 * ∇μ_ij * ∇σ²_ij'
    term5 = 0.75res_j * inv(varμ_j^2) * ∇σ²_ij * ∇σ²_ij'
    ∇²resβ_ij = term1 + term2 + term3 + term4 + term5

    return ∇²resβ_ij # p × p
end
function ∇²resβ_ij(dist, link, xj, z, η_j, μ_j, varμ_j, res_j)
    invσ_j = inv(sqrt(varμ_j))
    ∇ᵧμ_ij  = GLM.mueta(link, η_j) * z # 1 × 1
    ∇ᵦμ_ij  = GLM.mueta(link, η_j) * xj # p × 1
    ∇ᵧσ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * z # 1 × 1
    ∇ᵦσ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * xj # p × 1

    # assemble 5 terms
    term1 = -invσ_j * ∇²μ_j(link, η_j, xj, z)
    term2 = 0.5invσ_j^3 * ∇ᵦσ²_ij * ∇ᵧμ_ij
    term3 = -0.5 * res_j * inv(varμ_j) * ∇²σ²_j(dist, link, xj, z, μ_j, η_j)
    term4 = 0.5invσ_j^3 * ∇ᵦμ_ij * ∇ᵧσ²_ij
    term5 = 0.75res_j * inv(varμ_j^2) * ∇ᵦσ²_ij * ∇ᵧσ²_ij
    ∇²resβ_ij = term1 + term2 + term3 + term4 + term5

    return ∇²resβ_ij # p × 1
end

"""
    update_∇resβ(d::Distribution, x_ji, res_j, μ_j, dμ_j, varμ_j)

Computes ∇resβ_ij, the gradient of the standardized residuals for sample i 
at the j'th measurement. Here res_j is the standardized residual
"""
function update_∇resβ end
update_∇resβ(d::Normal, x_ji, res_j, μ_j, dμ_j, varμ_j) = -x_ji
update_∇resβ(d::Bernoulli, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -sqrt(varμ_j) * x_ji - (0.5 * res_j * (1 - 2μ_j) * x_ji)
update_∇resβ(d::Poisson, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    x_ji * (
    -(inv(sqrt(varμ_j)) + (0.5 * inv(varμ_j)) * res_j) * dμ_j
    )
update_∇resβ(d::NegativeBinomial, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -inv(sqrt(varμ_j)) * dμ_j * x_ji - (0.5 * inv(varμ_j)) *
    res_j * (μ_j * inv(d.r) + (1 + inv(d.r) * μ_j)) * dμ_j * x_ji
