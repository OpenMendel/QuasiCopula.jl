# Autoregressive AR(1) Covariance

In this notebook we will show how to form the quasi-copula model with Poisson base distribution, Log Link function and autoregressive AR(1) structured covariance on the `epilepsy` dataset provided in the `gcmr` R package. 

### Table of Contents:
* [Example 1: Intercept Only AR(1) Model](#Example-1:-Intercept-Only-AR(1)-Model)
* [Example 2: AR(1) Model with Covariates](#Example-2:-AR(1)-Model-with-Covariates)
* [Example 3: AR(1) Model with Covariates + L2 Penalty (optional)](#Example-3:-AR(1)-Model-with-Covariates-L2-penalty-(optional))

For these examples, we have $n$ independent clusters indexed by $i$. 

Under the AR(1) parameterization of the covariance matrix, the $i^{th}$ cluster with cluster size $d_i$, has covariance matrix $\mathbf{\Gamma_i}$ that takes the form: 

$$\mathbf{\Gamma_i}(\rho, \sigma^2) = \sigma^2 \times \left[\begin{array}{ccccccc}
1 & \rho & \rho^2 & \rho^3 & ...  &\rho^{d_i - 1}\\
 \rho & 1 & \rho & \rho^2 & ... \\
 & & ... & & \\ & &...& \rho & 1 & \rho \\
 \rho^{d_i - 1} & \rho^{d_i - 2} & ...& \rho^2 & \rho & 1
\end{array}\right]$$



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
using CSV, DataFrames, GLMCopula, LinearAlgebra, GLM, RCall
```

    ┌ Info: Precompiling GLMCopula [c47b6ae2-b804-4668-9957-eb588c99ffbc]
    └ @ Base loading.jl:1342



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


## Forming the Models

We can form the AR(1) models for regression with following arguments:

##### Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `covariates`: Covariate names of interest as a vector of `Symbol`s.
    Each variable name must be present in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

##### Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance parameter for the AR(1) structured covariance.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).

### Example 1: Intercept Only CS Model

We can form the AR(1) model with intercept only by excluding the `covariates` argument.


```julia
df = epilepsy
y = :counts
grouping = :id
d = Poisson()
link = LogLink()

Poisson_AR_model = AR_model(df, y, grouping, d, link)
```




    Quasi-Copula Autoregressive AR(1) Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 59
      * cluster size min, max: 5, 5
      * number of fixed effects: 1




### Example 2: AR(1) Model with Covariates

We can form the AR(1) model with covariates by including the `covariates` argument.


```julia
covariates = [:visit, :trt]

Poisson_AR_model = AR_model(df, y, grouping, covariates, d, link)
```




    Quasi-Copula Autoregressive AR(1) Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 59
      * cluster size min, max: 5, 5
      * number of fixed effects: 3




### Example 3: AR(1) Model with Covariates + L2 penalty (optional)

We can form the same AR(1) model from Example 2 with the optional argument for adding the L2 penalty on the variance parameter in the AR(1) parameterization of Gamma


```julia
Poisson_AR_model = AR_model(df, y, grouping, covariates, d, link; penalized = true)
```




    Quasi-Copula Autoregressive AR(1) Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 59
      * cluster size min, max: 5, 5
      * number of fixed effects: 3
      * L2 ridge penalty on AR(1) variance parameter: true



## Fitting the model

Let's show how to fit the model on the model from Example 3. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
GLMCopula.fit!(Poisson_AR_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing variance components using MM-Algorithm
    
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
    
    
    Number of Iterations....: 18
    
                                       (scaled)                 (unscaled)
    Objective...............:   5.2073621039210650e+02    2.1964303421468812e+03
    Dual infeasibility......:   2.3821081107655573e-08    1.0047571934396371e-07
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   1.0000000000000001e-11    4.2179328003577902e-11
    Overall NLP error.......:   2.3821081107655573e-08    1.0047571934396371e-07
    
    
    Number of objective function evaluations             = 24
    Number of objective gradient evaluations             = 19
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      2.958
    Total CPU secs in NLP function evaluations           =      0.008
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_AR_model.β
@show Poisson_AR_model.σ2
@show Poisson_AR_model.ρ;
```

    Poisson_AR_model.β = [3.474475582753216, -1.323384706329884, -0.04366460297735922]
    Poisson_AR_model.σ2 = [0.5348860843799086]
    Poisson_AR_model.ρ = [1.0]


Calculate the loglikelihood at the maximum


```julia
logl(Poisson_AR_model)
```




    -2196.4303424825557



Get asymptotic confidence intervals at the MLE's


```julia
get_CI(Poisson_AR_model)
```




    5×2 Matrix{Float64}:
      3.34736    3.60159
     -1.49822   -1.14855
     -0.489077   0.401748
      0.871632   1.12837
      0.461106   0.608666


