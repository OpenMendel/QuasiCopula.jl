# AR(1) Covariance Examples:

In this notebook we will fit our model on the two example datasets provided in the gcmr package:

    1. gcmr: Epilepsy 
    2. geepack: Respiratory

For these examples we will use the autoregressive AR(1) parameterization of the covariance matrix $\Gamma,$ estimating correlation parameter $\rho$ and dispersion parameter $\sigma^2$. 

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

We first demonstrate how to fit the model with Poisson base, and AR(1) covariance on the "epilepsy" dataset from the "gcmr" package in R.


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
epilepsy[1:10, :]
```




<div class="data-frame"><p>10 rows × 6 columns</p><table class="data-frame"><thead><tr><th></th><th>id</th><th>age</th><th>trt</th><th>counts</th><th>time</th><th>visit</th></tr><tr><th></th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>1</td><td>31</td><td>0</td><td>11</td><td>8.0</td><td>0.0</td></tr><tr><th>2</th><td>1</td><td>31</td><td>0</td><td>5</td><td>2.0</td><td>1.0</td></tr><tr><th>3</th><td>1</td><td>31</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr><tr><th>4</th><td>1</td><td>31</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr><tr><th>5</th><td>1</td><td>31</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr><tr><th>6</th><td>2</td><td>30</td><td>0</td><td>11</td><td>8.0</td><td>0.0</td></tr><tr><th>7</th><td>2</td><td>30</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr><tr><th>8</th><td>2</td><td>30</td><td>0</td><td>5</td><td>2.0</td><td>1.0</td></tr><tr><th>9</th><td>2</td><td>30</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr><tr><th>10</th><td>2</td><td>30</td><td>0</td><td>3</td><td>2.0</td><td>1.0</td></tr></tbody></table></div>



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

Fit the model


```julia
GLMCopula.fit!(Poisson_AR_model, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, hessian_approximation = "limited-memory"));
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing variance components using MM-Algorithm
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    This is Ipopt version 3.13.4, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).
    
    Number of nonzeros in equality constraint Jacobian...:        0
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:        0
    
    Total number of variables............................:        5
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  2.2098424e+03 0.00e+00 5.51e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  2.2101200e+03 0.00e+00 9.20e+01   1.2 2.84e+01    -  1.00e+00 7.38e-04f  7
       2  2.2078233e+03 0.00e+00 1.00e+02   1.1 2.34e-02    -  1.00e+00 1.00e+00f  1
       3  2.2009445e+03 0.00e+00 6.11e+01   0.9 1.28e-01    -  9.68e-01 1.00e+00f  1
       4  2.1819780e+03 0.00e+00 1.91e+02  -0.3 6.98e-01    -  1.00e+00 9.01e-01f  1
       5  2.1850781e+03 0.00e+00 1.17e+02   0.6 2.35e-01    -  1.00e+00 1.00e+00f  1
       6  2.1761316e+03 0.00e+00 5.85e+01   0.2 9.93e-02    -  1.00e+00 1.00e+00f  1
       7  2.1713613e+03 0.00e+00 6.13e+01  -0.4 1.69e-01    -  1.00e+00 1.00e+00f  1
       8  2.1715460e+03 0.00e+00 3.05e+01  -0.2 7.63e-02    -  1.00e+00 1.00e+00f  1
       9  2.1707466e+03 0.00e+00 1.53e+01  -0.4 2.97e-02    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.1705647e+03 0.00e+00 2.22e+01  -1.7 3.96e-02    -  1.00e+00 1.00e+00f  1
      11  2.1704662e+03 0.00e+00 5.39e+00  -1.1 1.94e-02    -  1.00e+00 1.00e+00f  1
      12  2.1704482e+03 0.00e+00 1.43e+00  -2.9 4.66e-03    -  1.00e+00 1.00e+00f  1
      13  2.1704468e+03 0.00e+00 2.80e-01  -4.8 1.67e-03    -  1.00e+00 1.00e+00f  1
      14  2.1704467e+03 0.00e+00 2.13e-01  -5.7 2.14e-04    -  1.00e+00 1.00e+00f  1
      15  2.1704465e+03 0.00e+00 3.28e-01  -7.6 1.34e-03    -  1.00e+00 1.00e+00f  1
      16  2.1704456e+03 0.00e+00 9.53e-01  -9.5 5.01e-03    -  1.00e+00 1.00e+00f  1
      17  2.1704450e+03 0.00e+00 1.40e+00 -10.6 2.03e-01    -  1.00e+00 6.25e-02f  5
      18  2.1704364e+03 0.00e+00 2.82e+00 -11.0 4.87e-02    -  1.00e+00 1.00e+00f  1
      19  2.1703854e+03 0.00e+00 6.27e+00 -11.0 3.08e-01    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      20  2.1703419e+03 0.00e+00 2.31e+01 -11.0 2.81e+00    -  1.00e+00 2.50e-01f  3
      21  2.1698936e+03 0.00e+00 1.24e+01 -11.0 4.02e+00    -  1.00e+00 1.00e+00f  1
      22  2.1697319e+03 0.00e+00 1.50e+01 -11.0 3.31e+00    -  1.00e+00 1.00e+00f  1
      23  2.1696644e+03 0.00e+00 2.57e+01 -10.8 2.57e+01    -  1.00e+00 1.25e-01f  4
      24  2.1696626e+03 0.00e+00 1.52e+01 -11.0 3.14e+00    -  1.00e+00 5.00e-01f  2
      25  2.1694235e+03 0.00e+00 1.63e+00 -11.0 1.02e+00    -  1.00e+00 1.00e+00f  1
      26  2.1693615e+03 0.00e+00 5.36e-01 -11.0 2.86e+00    -  1.00e+00 1.00e+00f  1
      27  2.1693610e+03 0.00e+00 1.42e+00 -10.5 5.96e+01    -  1.00e+00 9.80e-04f  7
      28  2.1692290e+03 0.00e+00 8.69e-01 -10.9 9.79e+00    -  1.00e+00 1.00e+00f  1
      29  2.1692198e+03 0.00e+00 8.65e+00 -11.0 1.24e+01    -  1.00e+00 2.50e-01f  3
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      30  2.1691272e+03 0.00e+00 2.78e+00 -11.0 1.26e+01    -  1.00e+00 1.00e+00f  1
      31  2.1691116e+03 0.00e+00 4.99e+00 -11.0 2.13e+01    -  1.00e+00 5.00e-01f  2
      32  2.1690512e+03 0.00e+00 1.07e+00 -11.0 1.40e+01    -  1.00e+00 1.00e+00f  1
      33  2.1690168e+03 0.00e+00 1.33e+00 -11.0 2.11e+01    -  1.00e+00 1.00e+00f  1
      34  2.1690104e+03 0.00e+00 3.53e+00 -11.0 7.18e+01    -  1.00e+00 2.50e-01f  3
      35  2.1689809e+03 0.00e+00 4.92e+00 -11.0 4.31e+01    -  1.00e+00 1.00e+00f  1
      36  2.1689670e+03 0.00e+00 2.03e+00 -10.3 6.53e+02    -  1.00e+00 3.12e-02f  6
      37  2.1689667e+03 0.00e+00 2.49e+00 -11.0 1.82e+02    -  1.00e+00 1.21e-02f  7
      38  2.1689533e+03 0.00e+00 7.65e-01 -11.0 2.83e+01    -  1.00e+00 1.00e+00f  1
      39  2.1689402e+03 0.00e+00 5.65e-01 -11.0 6.53e+01    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      40  2.1689319e+03 0.00e+00 1.33e+00 -11.0 7.85e+01    -  1.00e+00 1.00e+00f  1
      41  2.1689282e+03 0.00e+00 1.66e+00  -9.7 7.59e+03    -  4.88e-01 7.81e-03f  8
      42  2.1689248e+03 0.00e+00 4.06e+00 -11.0 1.14e+02    -  1.00e+00 1.00e+00f  1
      43  2.1689218e+03 0.00e+00 2.32e+00 -10.5 1.11e+03    -  1.00e+00 6.25e-02f  5
      44  2.1689194e+03 0.00e+00 2.73e+00 -11.0 2.13e+02    -  1.00e+00 5.00e-01f  2
      45  2.1689160e+03 0.00e+00 3.34e+00 -11.0 1.19e+02    -  1.00e+00 1.00e+00f  1
      46  2.1689116e+03 0.00e+00 5.29e-01 -11.0 5.59e+01    -  1.00e+00 1.00e+00f  1
      47  2.1689113e+03 0.00e+00 6.31e-01 -11.0 6.57e+01    -  1.00e+00 5.00e-01f  2
      48  2.1689096e+03 0.00e+00 3.05e+00  -9.0 3.76e+06    -  2.45e-04 1.63e-04f  7
      49  2.1689086e+03 0.00e+00 3.17e+00 -11.0 4.10e+02    -  1.00e+00 2.50e-01f  3
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      50  2.1689078e+03 0.00e+00 3.05e+00 -11.0 6.36e+02    -  1.00e+00 1.25e-01f  4
      51  2.1689049e+03 0.00e+00 2.20e-01 -11.0 2.23e+02    -  1.00e+00 1.00e+00f  1
      52  2.1689042e+03 0.00e+00 4.36e-01 -11.0 2.39e+02    -  1.00e+00 1.00e+00f  1
      53  2.1689028e+03 0.00e+00 4.31e-01 -11.0 7.12e+02    -  1.00e+00 1.00e+00f  1
      54  2.1689026e+03 0.00e+00 9.19e-01 -11.0 2.17e+03    -  1.00e+00 6.25e-02f  5
      55  2.1689015e+03 0.00e+00 5.24e-01 -11.0 1.09e+03    -  1.00e+00 1.00e+00f  1
      56  2.1689015e+03 0.00e+00 3.25e-01 -10.1 4.67e+05    -  1.83e-02 3.26e-05f  9
      57  2.1689014e+03 0.00e+00 4.19e-01 -10.3 4.13e+03    -  1.00e+00 3.12e-02f  6
      58  2.1689014e+03 0.00e+00 2.26e-01 -11.0 3.33e+02    -  1.00e+00 3.12e-02f  6
      59  2.1689007e+03 0.00e+00 5.59e-02 -11.0 1.20e+03    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      60  2.1689001e+03 0.00e+00 1.84e-02 -11.0 1.77e+03    -  1.00e+00 1.00e+00f  1
      61  2.1689001e+03 0.00e+00 1.11e-01 -10.8 1.04e+04    -  1.00e+00 3.91e-03f  9
      62  2.1689001e+03 0.00e+00 1.65e-03 -10.0 2.30e-03  -4.0 1.00e+00 1.00e+00f  1
      63  2.1688997e+03 0.00e+00 5.77e-03 -10.5 2.25e+03    -  1.00e+00 1.00e+00f  1
      64  2.1688995e+03 0.00e+00 3.16e-01 -11.0 3.04e+03    -  1.00e+00 1.00e+00f  1
      65  2.1688993e+03 0.00e+00 5.00e-01  -9.6 4.20e+05    -  5.94e-02 1.56e-02f  7
      66  2.1688991e+03 0.00e+00 1.06e-01 -11.0 1.49e+03    -  1.00e+00 1.00e+00f  1
      67  2.1688990e+03 0.00e+00 1.52e-01 -11.0 3.72e+03    -  1.00e+00 1.00e+00f  1
      68  2.1688990e+03 0.00e+00 2.80e-01  -9.8 1.58e+06    -  1.99e-02 1.95e-03f 10
      69  2.1688990e+03 0.00e+00 3.00e-01 -11.0 1.54e+05    -  1.00e+00 3.91e-03f  9
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      70  2.1688990e+03 0.00e+00 4.79e-01 -11.0 3.30e+03    -  1.00e+00 1.00e+00f  1
      71  2.1688989e+03 0.00e+00 1.32e-01 -11.0 9.69e-04  -4.5 1.00e+00 2.50e-01f  3
      72  2.1688989e+03 0.00e+00 2.21e-02 -11.0 3.47e+03    -  1.00e+00 1.00e+00f  1
      73  2.1688988e+03 0.00e+00 3.06e-01 -10.5 8.18e+04    -  1.00e+00 2.50e-01f  3
      74  2.1688987e+03 0.00e+00 5.11e-02 -11.0 1.04e+04    -  1.00e+00 1.00e+00f  1
      75  2.1688987e+03 0.00e+00 5.15e-02 -11.0 1.39e+04    -  1.00e+00 1.00e+00f  1
      76  2.1688987e+03 0.00e+00 5.45e-02 -11.0 2.70e-03  -5.0 1.00e+00 1.56e-02f  7
      77  2.1688987e+03 0.00e+00 5.12e-03 -11.0 3.14e-04  -5.4 1.00e+00 1.00e+00f  1
      78  2.1688987e+03 0.00e+00 1.91e-02 -11.0 1.72e+04    -  1.00e+00 1.00e+00f  1
    Cholesky factorization failed for LBFGS update! Skipping update.
      79  2.1688987e+03 0.00e+00 2.51e-02 -11.0 1.71e-03  -5.9 1.00e+00 1.56e-02f  7
    Cholesky factorization failed for LBFGS update! Skipping update.
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      80  2.1688987e+03 0.00e+00 2.18e-02 -11.0 4.55e-03  -6.4 1.00e+00 3.91e-03f  9
      81  2.1688987e+03 0.00e+00 2.10e-02 -11.0 9.26e-03  -6.9 1.00e+00 7.81e-03f  8
      82  2.1688987e+03 0.00e+00 1.17e-02 -11.0 2.10e-02    -  1.00e+00 2.44e-04f 13
      83  2.1688987e+03 0.00e+00 5.24e-03 -11.0 1.66e-06    -  1.00e+00 1.00e+00f  1
      84  2.1688987e+03 0.00e+00 4.70e-03 -11.0 2.59e-06    -  1.00e+00 1.00e+00f  1
      85  2.1688987e+03 0.00e+00 9.90e-04 -11.0 4.11e-06    -  1.00e+00 1.00e+00f  1
      86  2.1688987e+03 0.00e+00 2.83e-04 -11.0 7.16e-07    -  1.00e+00 1.00e+00f  1
      87  2.1688987e+03 0.00e+00 4.82e-05 -11.0 3.07e-07    -  1.00e+00 1.00e+00f  1
      88  2.1688987e+03 0.00e+00 1.96e-05 -11.0 6.12e-08    -  1.00e+00 1.00e+00f  1
      89  2.1688987e+03 0.00e+00 5.95e-08 -11.0 1.82e-08    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      90  2.1688987e+03 0.00e+00 1.21e-09 -11.0 2.20e-11    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 90
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.1688986516850264e+03    2.1688986516850264e+03
    Dual infeasibility......:   1.2101282776440716e-09    1.2101282776440716e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.9999999999999994e-12    9.9999999999999994e-12
    Overall NLP error.......:   1.2101282776440716e-09    1.2101282776440716e-09
    
    
    Number of objective function evaluations             = 381
    Number of objective gradient evaluations             = 91
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.642
    Total CPU secs in NLP function evaluations           =      0.028
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_AR_model.β
@show Poisson_AR_model.σ2
@show Poisson_AR_model.ρ;
```

    Poisson_AR_model.β = [3.477001434224112, -1.3123828083857931, -0.06552858740032046]
    Poisson_AR_model.σ2 = [96534.17924265226]
    Poisson_AR_model.ρ = [0.9499485377184236]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Poisson_AR_model, true, true);
```

    loglikelihood!(Poisson_AR_model, true, true) = -2168.8986516850264


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
respiratory_df[1:10, :]
```




<div class="data-frame"><p>10 rows × 8 columns</p><table class="data-frame"><thead><tr><th></th><th>center</th><th>id</th><th>treat</th><th>sex</th><th>age</th><th>baseline</th><th>visit</th><th>outcome</th></tr><tr><th></th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="CategoricalArrays.CategoricalValue{String, UInt32}">Cat…</th><th title="CategoricalArrays.CategoricalValue{String, UInt32}">Cat…</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>1</td><td>1</td><td>P</td><td>M</td><td>46</td><td>0</td><td>1</td><td>0</td></tr><tr><th>2</th><td>1</td><td>1</td><td>P</td><td>M</td><td>46</td><td>0</td><td>2</td><td>0</td></tr><tr><th>3</th><td>1</td><td>1</td><td>P</td><td>M</td><td>46</td><td>0</td><td>3</td><td>0</td></tr><tr><th>4</th><td>1</td><td>1</td><td>P</td><td>M</td><td>46</td><td>0</td><td>4</td><td>0</td></tr><tr><th>5</th><td>2</td><td>1</td><td>P</td><td>F</td><td>39</td><td>0</td><td>1</td><td>0</td></tr><tr><th>6</th><td>2</td><td>1</td><td>P</td><td>F</td><td>39</td><td>0</td><td>2</td><td>0</td></tr><tr><th>7</th><td>2</td><td>1</td><td>P</td><td>F</td><td>39</td><td>0</td><td>3</td><td>0</td></tr><tr><th>8</th><td>2</td><td>1</td><td>P</td><td>F</td><td>39</td><td>0</td><td>4</td><td>0</td></tr><tr><th>9</th><td>1</td><td>2</td><td>P</td><td>M</td><td>28</td><td>0</td><td>1</td><td>0</td></tr><tr><th>10</th><td>1</td><td>2</td><td>P</td><td>M</td><td>28</td><td>0</td><td>2</td><td>0</td></tr></tbody></table></div>



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

Fit the model


```julia
GLMCopula.fit!(Bernoulli_AR_model, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, hessian_approximation = "limited-memory"));
```

    initializing β using Newton's Algorithm under Independence Assumption
    initializing variance components using MM-Algorithm
    This is Ipopt version 3.13.4, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).
    
    Number of nonzeros in equality constraint Jacobian...:        0
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:        0
    
    Total number of variables............................:        6
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  2.5748718e+02 0.00e+00 6.94e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
    Warning: Cutting back alpha due to evaluation error
    Warning: Cutting back alpha due to evaluation error
    Warning: Cutting back alpha due to evaluation error
    Warning: Cutting back alpha due to evaluation error
    Warning: Cutting back alpha due to evaluation error
       1  2.5747386e+02 0.00e+00 5.37e+01   0.7 6.94e+01    -  1.00e+00 1.95e-05f 14
       2  2.5745540e+02 0.00e+00 1.71e+01  -1.0 5.71e-04    -  9.99e-01 1.00e+00f  1
       3  2.5745093e+02 0.00e+00 1.68e+01  -2.9 2.44e-04    -  1.00e+00 1.00e+00f  1
       4  2.5734241e+02 0.00e+00 6.40e+01  -4.3 6.94e-03    -  1.00e+00 1.00e+00f  1
       5  2.5650248e+02 0.00e+00 2.06e+02  -5.5 5.43e-02    -  1.00e+00 1.00e+00f  1
       6  2.5464997e+02 0.00e+00 6.92e+02  -5.6 1.03e+00    -  1.00e+00 7.13e-01f  1
       7  2.5418384e+02 0.00e+00 2.12e+02  -4.9 6.32e-01    -  1.00e+00 1.00e+00f  1
       8  2.5186151e+02 0.00e+00 2.23e+02  -5.8 9.11e-02    -  1.00e+00 1.00e+00f  1
       9  2.4931078e+02 0.00e+00 2.35e+02  -7.0 9.71e-02    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.4451164e+02 0.00e+00 2.51e+02  -4.1 1.23e+04    -  4.62e-05 1.82e-05f  2
      11  2.4395668e+02 0.00e+00 2.06e+02  -8.2 1.99e-01    -  1.00e+00 5.00e-01f  2
      12  2.4330412e+02 0.00e+00 1.20e+02  -6.2 4.48e-02    -  1.00e+00 1.00e+00f  1
      13  2.4292552e+02 0.00e+00 1.09e+01  -7.0 6.71e-02    -  1.00e+00 1.00e+00f  1
      14  2.4244163e+02 0.00e+00 3.33e+00  -8.2 1.69e-01    -  1.00e+00 1.00e+00f  1
      15  2.4206236e+02 0.00e+00 3.89e+01  -8.8 2.14e-01    -  1.00e+00 1.00e+00f  1
      16  2.4180565e+02 0.00e+00 3.00e+01 -10.5 2.91e-01    -  1.00e+00 1.00e+00f  1
      17  2.4162521e+02 0.00e+00 1.88e+01 -11.0 3.67e-01    -  1.00e+00 1.00e+00f  1
      18  2.4146266e+02 0.00e+00 1.53e+02 -11.0 4.09e-01    -  1.00e+00 1.00e+00f  1
      19  2.4119116e+02 0.00e+00 4.14e+01 -11.0 3.28e-01    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      20  2.4102867e+02 0.00e+00 7.80e+00 -11.0 4.92e-01    -  1.00e+00 1.00e+00f  1
      21  2.4100421e+02 0.00e+00 5.07e+01 -11.0 9.02e-01    -  1.00e+00 1.00e+00f  1
      22  2.4100048e+02 0.00e+00 7.62e+00 -11.0 1.15e+00    -  1.00e+00 1.00e+00f  1
      23  2.4086574e+02 0.00e+00 1.43e+01 -11.0 5.22e-01    -  1.00e+00 1.00e+00f  1
      24  2.4084305e+02 0.00e+00 1.46e+01 -11.0 2.20e-01    -  1.00e+00 1.00e+00f  1
      25  2.4083266e+02 0.00e+00 9.55e+00 -10.5 1.23e+01    -  1.00e+00 3.02e-02f  5
      26  2.4077281e+02 0.00e+00 3.02e+00 -11.0 9.77e-01    -  1.00e+00 1.00e+00f  1
      27  2.4077219e+02 0.00e+00 9.18e-01 -10.4 1.53e+01    -  1.00e+00 6.29e-04f  8
      28  2.4073360e+02 0.00e+00 1.05e+00 -11.0 1.31e+00    -  1.00e+00 1.00e+00f  1
      29  2.4069541e+02 0.00e+00 1.29e+01 -11.0 2.47e+00    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      30  2.4069365e+02 0.00e+00 4.26e+01  -9.0 5.10e+04    -  1.93e-04 2.39e-05f  6
      31  2.4065211e+02 0.00e+00 3.24e+00 -11.0 2.13e+00    -  1.00e+00 1.00e+00f  1
      32  2.4063492e+02 0.00e+00 7.37e+00 -11.0 2.90e+00    -  1.00e+00 1.00e+00f  1
      33  2.4063487e+02 0.00e+00 1.29e+01 -10.8 2.39e+01    -  1.00e+00 1.56e-02f  7
      34  2.4062694e+02 0.00e+00 3.01e+00 -11.0 9.09e+00    -  1.00e+00 5.00e-01f  2
      35  2.4060280e+02 0.00e+00 5.94e+00 -10.6 3.94e+01    -  1.00e+00 2.50e-01f  3
      36  2.4060265e+02 0.00e+00 3.12e+00 -11.0 1.05e+01    -  1.00e+00 1.56e-02f  7
      37  2.4059292e+02 0.00e+00 3.93e+00 -11.0 3.94e+00    -  1.00e+00 1.00e+00f  1
      38  2.4058555e+02 0.00e+00 7.68e-01 -11.0 8.57e+00    -  1.00e+00 1.00e+00f  1
      39  2.4058489e+02 0.00e+00 7.02e+00  -9.0 5.74e+03    -  3.48e-02 4.88e-04f 12
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      40  2.4058476e+02 0.00e+00 1.00e+01 -11.0 6.45e+01    -  1.00e+00 3.12e-02f  6
      41  2.4057821e+02 0.00e+00 4.90e+00 -11.0 1.37e+01    -  1.00e+00 1.00e+00f  1
      42  2.4057663e+02 0.00e+00 1.00e+01 -10.4 2.19e+02    -  1.00e+00 3.12e-02f  6
      43  2.4057247e+02 0.00e+00 3.30e+00 -11.0 1.18e+01    -  1.00e+00 1.00e+00f  1
      44  2.4056879e+02 0.00e+00 1.46e+00 -11.0 2.45e+01    -  1.00e+00 1.00e+00f  1
      45  2.4056629e+02 0.00e+00 2.22e+00 -11.0 2.92e+01    -  1.00e+00 1.00e+00f  1
      46  2.4056629e+02 0.00e+00 1.14e+00 -10.7 2.34e+02    -  1.00e+00 9.77e-04f 11
      47  2.4056597e+02 0.00e+00 1.96e+00 -11.0 4.82e+01    -  1.00e+00 2.50e-01f  3
      48  2.4056538e+02 0.00e+00 2.88e-01 -11.0 6.94e-01  -4.0 1.00e+00 1.00e+00f  1
      49  2.4056483e+02 0.00e+00 3.51e+00  -9.3 6.54e+03    -  1.00e+00 2.50e-01f  3
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      50  2.4056453e+02 0.00e+00 1.91e+00  -9.2 1.80e+00  -4.5 1.00e+00 7.81e-03f  8
      51  2.4055821e+02 0.00e+00 3.27e+00 -11.0 4.53e+01    -  1.00e+00 1.00e+00f  1
      52  2.4055818e+02 0.00e+00 6.80e+00 -10.2 5.39e+04    -  3.37e-02 3.12e-02f  6
      53  2.4055796e+02 0.00e+00 3.85e+00 -11.0 4.34e-02  -5.0 1.00e+00 5.00e-01f  2
      54  2.4055743e+02 0.00e+00 2.59e-01 -11.0 3.98e+02    -  1.00e+00 1.00e+00f  1
      55  2.4055743e+02 0.00e+00 9.86e-02 -11.0 2.17e-02  -5.4 1.00e+00 2.50e-01f  3
      56  2.4055738e+02 0.00e+00 2.08e-01 -11.0 5.30e+02    -  1.00e+00 1.00e+00f  1
      57  2.4055738e+02 0.00e+00 2.37e-01 -11.0 4.12e-02  -5.9 1.00e+00 1.25e-01f  4
      58  2.4055731e+02 0.00e+00 1.76e-01 -11.0 1.66e+03    -  1.00e+00 1.00e+00f  1
      59  2.4055731e+02 0.00e+00 1.29e-01 -11.0 1.34e-01  -6.4 1.00e+00 3.91e-03f  9
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      60  2.4055731e+02 0.00e+00 2.05e-01 -11.0 8.30e+03    -  7.25e-01 3.12e-02f  6
      61  2.4055731e+02 0.00e+00 2.79e-02 -11.0 2.46e-01  -6.9 1.00e+00 9.77e-04f 11
      62  2.4055725e+02 0.00e+00 5.85e-03 -11.0 1.85e+03    -  1.00e+00 1.00e+00f  1
      63  2.4055725e+02 0.00e+00 5.79e-03 -11.0 3.61e-01  -7.3 1.00e+00 1.95e-03f 10
      64  2.4055724e+02 0.00e+00 1.74e-01 -11.0 2.29e+04    -  3.54e-01 6.25e-02f  5
      65  2.4055723e+02 0.00e+00 3.80e-02 -11.0 1.25e+00  -7.8 1.00e+00 1.00e+00f  1
      66  2.4055713e+02 0.00e+00 7.19e-02 -10.3 4.29e+04    -  1.00e+00 1.00e+00f  1
      67  2.4055713e+02 0.00e+00 1.93e-01 -11.0 1.98e+00  -8.3 1.00e+00 1.95e-03f 10
      68  2.4055713e+02 0.00e+00 1.16e-03 -11.0 1.47e+03    -  1.00e+00 1.00e+00f  1
      69  2.4055713e+02 0.00e+00 1.83e-04 -11.0 1.39e+03    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      70  2.4055713e+02 0.00e+00 2.74e-04 -11.0 2.38e-01  -8.8 1.00e+00 4.88e-04f 12
      71  2.4055712e+02 0.00e+00 3.78e-05 -11.0 2.68e+04    -  1.00e+00 1.00e+00f  1
      72  2.4055712e+02 0.00e+00 3.01e-04 -11.0 1.07e+00  -9.7 1.00e+00 2.44e-04f 13
      73  2.4055712e+02 0.00e+00 1.16e-04 -11.0 2.22e+04    -  1.00e+00 1.00e+00f  1
      74  2.4055712e+02 0.00e+00 2.18e-04 -11.0 8.35e-02  -8.9 1.00e+00 2.44e-04f 13
      75  2.4055711e+02 0.00e+00 1.10e-04 -11.0 3.62e+04    -  1.00e+00 1.00e+00f  1
      76  2.4055711e+02 0.00e+00 4.24e-02 -11.0 3.95e+04    -  1.00e+00 1.00e+00f  1
      77  2.4055711e+02 0.00e+00 2.04e-02 -11.0 2.49e-01  -9.8 1.00e+00 3.91e-03f  9
      78  2.4055711e+02 0.00e+00 3.25e-03 -11.0 6.45e+04    -  1.00e+00 1.00e+00f  1
      79  2.4055711e+02 0.00e+00 1.71e-03 -11.0 3.39e+04    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      80  2.4055711e+02 0.00e+00 1.43e-03 -11.0 2.84e+04    -  1.00e+00 1.00e+00f  1
      81  2.4055711e+02 0.00e+00 1.39e-03 -11.0 1.43e-03    -  1.00e+00 3.05e-05f 16
      82  2.4055711e+02 0.00e+00 3.35e-05 -11.0 2.15e-08    -  1.00e+00 1.00e+00f  1
      83  2.4055711e+02 0.00e+00 3.34e-05 -11.0 5.22e-10    -  1.00e+00 1.00e+00f  1
      84  2.4055711e+02 0.00e+00 1.39e-04 -11.0 1.93e-07    -  1.00e+00 1.00e+00f  1
      85  2.4055711e+02 0.00e+00 3.89e-04 -11.0 3.69e-07    -  1.00e+00 1.00e+00f  1
      86  2.4055711e+02 0.00e+00 6.71e-04 -11.0 1.27e-06    -  1.00e+00 1.00e+00f  1
      87  2.4055711e+02 0.00e+00 5.33e-04 -11.0 1.19e-06    -  1.00e+00 1.00e+00f  1
      88  2.4055711e+02 0.00e+00 1.01e-04 -11.0 5.35e-07    -  1.00e+00 1.00e+00f  1
      89  2.4055711e+02 0.00e+00 1.78e-05 -11.0 3.15e-07    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      90  2.4055711e+02 0.00e+00 1.48e-05 -11.0 6.35e-07    -  1.00e+00 1.00e+00f  1
      91  2.4055711e+02 0.00e+00 5.19e-05 -11.0 2.40e-07    -  1.00e+00 1.00e+00f  1
      92  2.4055711e+02 0.00e+00 1.74e-06 -11.0 1.68e-07    -  1.00e+00 1.00e+00f  1
      93  2.4055711e+02 0.00e+00 1.97e-07 -11.0 6.07e-10    -  1.00e+00 1.00e+00f  1
      94  2.4055711e+02 0.00e+00 1.41e-09 -11.0 2.06e-10    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 94
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.4055710825065927e+02    2.4055710825065927e+02
    Dual infeasibility......:   1.4119798663614347e-09    1.4119798663614347e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   1.0000000000000001e-11    1.0000000000000001e-11
    Overall NLP error.......:   1.4119798663614347e-09    1.4119798663614347e-09
    
    
    Number of objective function evaluations             = 421
    Number of objective gradient evaluations             = 95
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.698
    Total CPU secs in NLP function evaluations           =      0.027
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Bernoulli_AR_model.β
@show Bernoulli_AR_model.σ2
@show Bernoulli_AR_model.ρ;
```

    Bernoulli_AR_model.β = [-0.858664409049024, 0.8334076581881305, -0.026953129746342567, 2.103267661442157]
    Bernoulli_AR_model.σ2 = [306890.7562627383]
    Bernoulli_AR_model.ρ = [0.7813892966990003]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Bernoulli_AR_model, true, true);
```

    loglikelihood!(Bernoulli_AR_model, true, true) = -240.55710825065927

