# SymmetricEigenProblem.jl

A more or less working implementation of the QR-algorithm for real, symmetric, tridiagonal matrices.

## Features
- Fast computation of eigenvectors through applying Given's rotations in bulk without using GEMM. Assumes avx2 currently, but can be made generic. [1]
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


## Important note

Because this is a prototype, it only supports Float64 SymTridiagonal matrices which have an **order divisable by 4**. It's trivial to support many more types of matrices, but that's for later. If your matrix does not have this order, julia might segfault :).

## Stability issues

Sometimes the algorithm might use exact eigenvalues as shifts, which might introduce extremely big errors. I have to figure out how to stabilize it a bit more, but it should be fine as longs as the eigenvalues you have are separated a bit.

## Example

```julia
julia> using SymmetricEigenProblem

julia> n = 2000;

julia> A = SymTridiagonal(collect(1.0 : n), rand(n - 1));

julia> Q = Matrix(1.0I, n, n);

julia> D = copy(A)

julia> qr_algorithm!(D, Q);

julia> norm(A * Q - Q * Diagonal(D))
2.8632552826583774e-10
```

[1] Van Zee, Field G., Robert A. Van De Geijn, and Gregorio Quintana-Orti. "Restructuring the QR-Algorithm for High-Performance Applications of Givens Rotations." (2011).
