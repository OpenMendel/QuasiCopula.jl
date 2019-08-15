__precompile__()

module GLMCopula

using Convex, LinearAlgebra, MathProgBase, Reexport
@reexport using Ipopt
@reexport using NLopt

include("gaussian_vc.jl")

end#module