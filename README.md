# SymmetricEigenProblem.jl

A more or less working implementation of the QR-algorithm for real, symmetric, tridiagonal matrices.

## Features
- Fast computation of eigenvectors through bulk Given's rotation without using GEMM. Assumes avx2 currently, but can be made generic. [1]
- Otherwise standard bulge chasing with Wilkinson shifts, no multiple tightly packed bulges or anything.

The Given's rotations can theoretically get 75% of dgemm performance (4 muls, 2 adds). To benchmark this non-blas-type routine, try:

```julia
julia> include("benchmark/benchmark.jl")

julia> bench(2000, 2000, 64)
45.180626626219976
```

which will apply `64 × 1999` Given's rotations to the columns of a `2000 × 2000` matrix and output the GFLOP/s. On my computer single-threaded peakflops is:

```julia
julia> using LinearAlgebra

julia> BLAS.set_num_threads(1)

julia> LinearAlgebra.peakflops() / 1e9
60.28352275144608
```

So indeed roughly `74.95%` of dgemm.


[1] Van Zee, Field G., Robert A. Van De Geijn, and Gregorio Quintana-Orti. "Restructuring the QR-Algorithm for High-Performance Applications of Givens Rotations." (2011).
