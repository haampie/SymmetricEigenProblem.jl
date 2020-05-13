# SymmetricEigenProblem.jl

A more or less working implementation of the QR-algorithm for real, symmetric, tridiagonal matrices.

Turns out trivial bulge-chasing can be competitive with MRRR.

## Features
- Fast computation of eigenvectors through applying Given's rotations in bulk without using GEMM. [1]
- Otherwise standard bulge chasing with Wilkinson shifts, no multiple tightly packed bulges or anything.

The Given's rotations can theoretically get 75% of gemm performance (4 muls, 2 adds). To benchmark this non-blas-type routine, try:

```julia
julia> include("benchmark/benchmark.jl")

julia> bench(Float64, 2000, 2000, 64)
45.722959723206735

julia> bench(Float32, 2000, 2000, 64)
89.314389055655
```

which will apply `64 × 1999` Given's rotations to the columns of a `2000 × 2000` matrix and output the GFLOP/s. On my computer single-threaded peakflops is:

```julia
julia> using LinearAlgebra

julia> BLAS.set_num_threads(1)

julia> LinearAlgebra.peakflops() / 1e9
60.28352275144608
```

So indeed roughly `74.95%` of dgemm.

## Threading

For threading use `JULIA_NUM_THREADS=8 julia`. It will parellellize the application of Given's rotations and gives significant speedups.

```julia
julia> include("benchmark/benchmark.jl")

julia> bench_parallel(Float64, 2000, 2000, 64)
267.1348970050438

julia> bench_parallel(Float32, 2000, 2000, 64)
530.8279734148967
```

## Caveats

Currently only supports matrices of order divisable by 2. 

This restriction can be lifted when the fused Given's rotation kernels are generated (4 rotations are combined, so ≤ 16 different kernels, I've implemented 3 by hand in `src/bulk_givens.jl`).

## Stability issues

Sometimes the algorithm might use exact eigenvalues as shifts, which might introduce extremely big errors. I have to figure out how to stabilize it a bit more, but it should be fine as longs as the eigenvalues you have are separated a bit.

## Example

```julia
julia> using LinearAlgebra, SymmetricEigenProblem

julia> n = 2000;

julia> A = SymTridiagonal(collect(1.0 : n), rand(n - 1));

julia> Q = Matrix(1.0I, n, n);

julia> D = copy(A);

julia> @time SymmetricEigenProblem.qr_algorithm!(D, Q);
  0.258179 seconds (1.15 k allocations: 4.041 MiB)

julia> norm(A * Q - Q * Diagonal(D))
2.8632552826583774e-10
```

[1] Van Zee, Field G., Robert A. Van De Geijn, and Gregorio Quintana-Orti. "Restructuring the QR-Algorithm for High-Performance Applications of Givens Rotations." (2011).
