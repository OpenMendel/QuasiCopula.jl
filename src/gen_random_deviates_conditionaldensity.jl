### joint and conditional density
using Roots
function joint_density_value(density::D, res::Vector{T}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    pdf_vector = pdf.(density, res)
    joint_pdf = 1.0
    for i in 1:length(pdf_vector)
        joint_pdf = pdf_vector[i] * joint_pdf
    end
    return joint_pdf
 end
 
 # this function will get the cross terms for s, and all the cross terms up to s if all = true; used in marginal density of i in S
 function crossterm_res(res::Vector{T}, s::Integer, Γ::Matrix{T}; all = false) where {T<: BlasReal}
    results = []
    if s == 1
        return 0.0
    elseif s > 1
        if all == true
            for i in 2:s
                for j in 1:i - 1
                    push!(results, res[i] * sum(res[j] * Γ[i, j]))
                end
            end
        else
            for j in 1:s - 1
                push!(results, res[s] * sum(res[j] * Γ[s, j]))
            end
        end
    end
    return results
 end

###
# now we want to look at the way the root finding package works, and define the conditional density of Ri | R1, ..., Ri-1 as a function of x = R_i.

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
## currently for normal density it is using the residuals to simulate the vectors.

# function conditional_terms_ken!(gvc_vec::GVCVec{T, D}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
#     for i in 1:gvc_vec.n
#         gvc_vec.term1[i] = 1 + 0.5 * transpose( gvc_vec.res[1:i-1]) *  gvc_vec.Γ[1:i-1, 1:i-1] *  gvc_vec.res[1:i-1] +  0.5 * tr( gvc_vec.Γ[i:end, i:end])
#         gvc_vec.term2[i] = sum(crossterm_res( gvc_vec.res, i, gvc_vec.Γ))
#         fun(x) = (0.5 * gvc_vec.Γ[i, i] * (x^2 - 1))
#         gvc_vec.term3[i] = fun
#         fun2(x) = inv(gvc_vec.term1[i]) * pdf(gvc_vec.vecd[i], x) * (gvc_vec.term1[i] +  gvc_vec.term2[i] +  gvc_vec.term3[i](x))
#         gvc_vec.conditional_pdf[i] = fun2
#     end
# end

function conditional_terms!(gvc_vec::GVCVec{T, D}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution} 
    for i in 1:gvc_vec.n
        gvc_vec.term1[i] = 1 + 0.5 * transpose( gvc_vec.res[1:i-1]) *  gvc_vec.Γ[1:i-1, 1:i-1] *  gvc_vec.res[1:i-1] +  0.5 * tr(gvc_vec.Γ[i+1:end, i+1:end])
        gvc_vec.term2[i] = sum(crossterm_res( gvc_vec.res, i, gvc_vec.Γ))
        fun(x) = 0.5 * gvc_vec.Γ[i, i] * x^2
        gvc_vec.term3[i] = fun
    end
end

function conditional_pdf_cdf!(gvc_vec::GVCVec{T, D}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution} 
    for i in 1:gvc_vec.n
        fun2(x) = inv(1 + 0.5 * gvc_vec.trΓ) * pdf(gvc_vec.vecd[i], x) * (gvc_vec.term1[i] +  gvc_vec.term2[i] +  gvc_vec.term3[i](x))
        gvc_vec.conditional_pdf[i] = fun2
        if i == 1
            gvc_vec.rsminus1_γis[i] = 0.0
        else
            gvc_vec.rsminus1_γis[i] = sum(gvc_vec.res[j] * gvc_vec.Γ[i, j] for j in 1:i-1)
        end
        fun3(x) = inv(1 + 0.5 * gvc_vec.trΓ) * [(gvc_vec.term1[i]) * cdf(gvc_vec.vecd[i], x) - gvc_vec.rsminus1_γis[i] * gvc_vec.conditional_pdf[i](x) + 0.25 * gvc_vec.Γ[i, i] * (1 + sign(x) * cdf(Chisq(3), x^2))]
        gvc_vec.conditional_cdf[i] = fun3
    end
end

# This function is the user interface on object. Recursively fills up the residual vector.
function generate_res_vec!(gvc_vec::GVCVec{T, D}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    for i in 1:gvc_vec.n
        conditional_terms!(gvc_vec) 
        conditional_pdf_cdf!(gvc_vec)
        gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
        F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
        gvc_vec.res[i] = find_zero(F_r, (-50, 50), Roots.Bisection())
    end
end
