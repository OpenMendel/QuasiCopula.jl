# Compound Symmetric (CS) Covariance 

In this notebook we will fit our model on the two example datasets provided in the geepack ang gcmr R packages. For these examples we will use the compound symmetry (CS) parameterization of the covariance matrix $\Gamma,$ estimating correlation parameter $\rho$ and dispersion parameter $\sigma^2$. 

### Table of Contents:
* [Example 1: Poisson CS (gcmr:Epilepsy)](#Example-1:-Poisson-CS)
* [Example 2: Bernoulli CS (geepack:Respiratory)](#Example-2:-Bernoulli-CS)

    note: For the dispersion parameter, we can an L2 penalty to the loglikelihood to keep the estimates from going off to infinity. This notebook presents results with the unpenalized fit.


```julia
versioninfo()
```

    Julia Version 1.6.2
    Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin18.7.0)
      CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-11.0.1 (ORCJIT, skylake)



```julia
using CSV, DataFrames, GLMCopula, LinearAlgebra, GLM, RCall, RData, RDatasets
```

## Example 1: Poisson CS

We first demonstrate how to fit the model with Poisson base and CS covariance on the "epilepsy" dataset from the "gcmr" package in R.


```julia
R"""
    library("gcmr")
    data("epilepsy", package = "gcmr")
"""
@rget epilepsy;
```

    ┌ Warning: RCall.jl: Warning: package ‘gcmr’ was built under R version 4.0.2
    └ @ RCall /Users/sarahji/.julia/packages/RCall/6kphM/src/io.jl:172


Let's take a preview of the first 10 lines of the epilepsy dataset.


```julia
@show epilepsy[1:10, :];
```

    epilepsy[1:10, :] = 10×6 DataFrame
     Row │ id     age    trt    counts  time     visit
         │ Int64  Int64  Int64  Int64   Float64  Float64
    ─────┼───────────────────────────────────────────────
       1 │     1     31      0      11      8.0      0.0
       2 │     1     31      0       5      2.0      1.0
       3 │     1     31      0       3      2.0      1.0
       4 │     1     31      0       3      2.0      1.0
       5 │     1     31      0       3      2.0      1.0
       6 │     2     30      0      11      8.0      0.0
       7 │     2     30      0       3      2.0      1.0
       8 │     2     30      0       5      2.0      1.0
       9 │     2     30      0       3      2.0      1.0
      10 │     2     30      0       3      2.0      1.0


To form the model, we give it the following arguments:

- named dataframe
- outcome variable name of interest as a symbol
- grouping variable name of interest as a symbol
- covariate names of interest as a vector of symbols
- base distribution
- link function



```julia
df = epilepsy
y = :counts
grouping = :id
covariates = [:visit, :trt]
d = Poisson()
link = LogLink()

Poisson_CS_model = CS_model(df, y, grouping, covariates, d, link);
```

Fit the model


```julia
GLMCopula.fit!(Poisson_CS_model, IpoptSolver(print_level = 3, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, hessian_approximation = "limited-memory"));
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing σ2 and ρ using method of moments
    par0 = [3.4553888182001127, -1.3288345481797776, -0.026384675366883565, 0.2, 1.0]
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    Total number of variables............................:        5
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 92
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.1695813266189330e+03    2.1695813266189330e+03
    Dual infeasibility......:   9.1969454274476448e-09    9.1969454274476448e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.9999999999999994e-12    9.9999999999999994e-12
    Overall NLP error.......:   9.1969454274476448e-09    9.1969454274476448e-09
    
    
    Number of objective function evaluations             = 379
    Number of objective gradient evaluations             = 93
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.589
    Total CPU secs in NLP function evaluations           =      0.024
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_CS_model.β
@show Poisson_CS_model.σ2
@show Poisson_CS_model.ρ;
```

    Poisson_CS_model.β = [3.479229893235856, -1.3137359424301136, -0.05223916238672295]
    Poisson_CS_model.σ2 = [133560.55732336888]
    Poisson_CS_model.ρ = [0.9101785804195232]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Poisson_CS_model, true, true);
```

    loglikelihood!(Poisson_CS_model, true, true) = -2169.581326618933


## Example 2: Bernoulli CS

We will next demo how to fit the model with Bernoulli base and CS covariance on the "respiratory" dataset from the "geepack" package. 


```julia
R"""
    data(respiratory, package="geepack")
    respiratory_df <- respiratory[order(respiratory$id),]
"""

@rget respiratory_df;
```

Let's take a preview of the first 10 lines of the respiratory dataset in long format.


```julia
@show respiratory_df[1:10, :];
```

    respiratory_df[1:10, :] = 10×8 DataFrame
     Row │ center  id     treat  sex   age    baseline  visit  outcome
         │ Int64   Int64  Cat…   Cat…  Int64  Int64     Int64  Int64
    ─────┼─────────────────────────────────────────────────────────────
       1 │      1      1  P      M        46         0      1        0
       2 │      1      1  P      M        46         0      2        0
       3 │      1      1  P      M        46         0      3        0
       4 │      1      1  P      M        46         0      4        0
       5 │      2      1  P      F        39         0      1        0
       6 │      2      1  P      F        39         0      2        0
       7 │      2      1  P      F        39         0      3        0
       8 │      2      1  P      F        39         0      4        0
       9 │      1      2  P      M        28         0      1        0
      10 │      1      2  P      M        28         0      2        0


To form the model, we give it the following arguments:

- named dataframe
- outcome variable name of interest as a symbol
- grouping variable name of interest as a symbol
- covariate names of interest as a vector of symbols
- base distribution
- link function


```julia
df = respiratory_df
y = :outcome
grouping = :id
covariates = [:center, :age, :baseline]
d = Bernoulli()
link = LogitLink()

Bernoulli_CS_model = CS_model(df, y, grouping, covariates, d, link);
```

Fit the model


```julia
GLMCopula.fit!(Bernoulli_CS_model, IpoptSolver(print_level = 3, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, hessian_approximation = "limited-memory"));
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing σ2 and ρ using method of moments
    par0 = [-0.7993115972643741, 0.6513519744878128, -0.018744798735221512, 1.6766993967179145, 0.2, 1.0]
    Total number of variables............................:        6
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 27
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.4900152024121175e+02    2.4900152024121175e+02
    Dual infeasibility......:   8.5782119185751071e-09    8.5782119185751071e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   1.0000000000000003e-11    1.0000000000000003e-11
    Overall NLP error.......:   8.5782119185751071e-09    8.5782119185751071e-09
    
    
    Number of objective function evaluations             = 46
    Number of objective gradient evaluations             = 28
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.134
    Total CPU secs in NLP function evaluations           =      0.010
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Bernoulli_CS_model.β
@show Bernoulli_CS_model.σ2
@show Bernoulli_CS_model.ρ;
```

    Bernoulli_CS_model.β = [-0.8073560280686407, 0.8553513879813671, -0.027821756706670475, 2.0702779503223048]
    Bernoulli_CS_model.σ2 = [0.35242279695900536]
    Bernoulli_CS_model.ρ = [0.8734724569746284]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Bernoulli_CS_model, true, true);
```

    loglikelihood!(Bernoulli_CS_model, true, true) = -249.00152024121175

