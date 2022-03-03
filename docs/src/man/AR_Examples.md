# Autoregressive AR(1) Covariance

In this notebook we will fit our model with autoregressive AR(1) structured covariance on two example datasets provided in the gcmr and geepack R packages. 

For these examples, we have $n$ independent clusters indexed by $i$. 

Under the AR(1) parameterization of the covariance matrix, the $i^{th}$ cluster with cluster size $d_i$, has covariance matrix $\mathbf{\Gamma_i}$ that takes the form: 

$$\mathbf{\Gamma_i}(\rho, \sigma^2) = \sigma^2 \times \left[\begin{array}{ccccccc}
1 & \rho & \rho^2 & \rho^3 & ...  &\rho^{d_i - 1}\\
 \rho & 1 & \rho & \rho^2 & ... \\
 & & ... & & \\ & &...& \rho & 1 & \rho \\
 \rho^{d_i - 1} & \rho^{d_i - 2} & ...& \rho^2 & \rho & 1
\end{array}\right]$$

### Table of Contents:
* [Example 1: Poisson AR(1) (gcmr:Epilepsy)](#Example-1:-Poisson-AR(1))
* [Example 2: Bernoulli AR(1) (geepack:Respiratory)](#Example-2:-Bernoulli-AR(1))

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

## Example 1: Poisson AR(1) 

We first demonstrate how to fit the model with Poisson base and AR(1) covariance on the "epilepsy" dataset from the "gcmr" package in R.


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


#### Forming the Model

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

Poisson_AR_model = AR_model(df, y, grouping, covariates, d, link);
```

Fit the model. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


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
    
    
    Number of Iterations....: 89
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.1688986516850264e+03    2.1688986516850264e+03
    Dual infeasibility......:   5.9531302554205467e-08    5.9531302554205467e-08
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.9999999999999994e-12    9.9999999999999994e-12
    Overall NLP error.......:   5.9531302554205467e-08    5.9531302554205467e-08
    
    
    Number of objective function evaluations             = 380
    Number of objective gradient evaluations             = 90
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.600
    Total CPU secs in NLP function evaluations           =      0.023
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_AR_model.β
@show Poisson_AR_model.σ2
@show Poisson_AR_model.ρ;
```

    Poisson_AR_model.β = [3.477001434202149, -1.3123828083769642, -0.06552858739753331]
    Poisson_AR_model.σ2 = [96534.17924265226]
    Poisson_AR_model.ρ = [0.949948537714978]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Poisson_AR_model, false, false);
```

    loglikelihood!(Poisson_AR_model, false, false) = -2168.8986516850264


## Example 2: Bernoulli AR(1)

We will next demo how to fit the model with Bernoulli base and AR(1) covariance on the "respiratory" dataset from the "geepack" package. 


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


#### Forming the Model

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

Bernoulli_AR_model = AR_model(df, y, grouping, covariates, d, link);
```

Fit the model. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
GLMCopula.fit!(Bernoulli_AR_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing variance components using MM-Algorithm
    Total number of variables............................:        6
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 93
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.4055710825065918e+02    2.4055710825065918e+02
    Dual infeasibility......:   1.9741609591505949e-07    1.9741609591505949e-07
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.9999999999999994e-12    9.9999999999999994e-12
    Overall NLP error.......:   1.9741609591505949e-07    1.9741609591505949e-07
    
    
    Number of objective function evaluations             = 420
    Number of objective gradient evaluations             = 94
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.661
    Total CPU secs in NLP function evaluations           =      0.021
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Bernoulli_AR_model.β
@show Bernoulli_AR_model.σ2
@show Bernoulli_AR_model.ρ;
```

    Bernoulli_AR_model.β = [-0.8586644089629349, 0.8334076580604213, -0.02695312973934001, 2.1032676612358934]
    Bernoulli_AR_model.σ2 = [306890.7562627383]
    Bernoulli_AR_model.ρ = [0.7813892966698839]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Bernoulli_AR_model, false, false);
```

    loglikelihood!(Bernoulli_AR_model, false, false) = -240.55710825065918

