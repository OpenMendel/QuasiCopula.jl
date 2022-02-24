using Documenter
using GLMCopula

makedocs(
    sitename = "GLMCopula",
    format = Documenter.HTML(),
    modules = [GLMCopula],
    pages = [
        "Home" => "index.md",
        "API" => "man/api.md",
        "AR(1) Covariance" => "man/AR_Examples.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo   = "github.com/sarah-ji/GLMCopula.jl.git",
    target = "build"
)
