using SymmetricEigenProblem
using BenchmarkTools

function bench(T = Float64, m = 2000, n = 2000, k = 64)

    givens = fill((rand(T), rand(T)), n - 1, k)

    Q = rand(T, m, n)

    # 6 flops per givens rotation per row
    flops = 6 * k * (n - 1) * m

    time = @belapsed SymmetricEigenProblem.bulk_wave_order_2x2_rmul!($Q, $givens)

    return flops / time / 1e9
end

function bench_parallel(T = Float64, m = 2000, n = 2000, k = 64)

    givens = fill((rand(T), rand(T)), n - 1, k)

    Q = rand(T, m, n)

    # 6 flops per givens rotation per row
    flops = 6 * k * (n - 1) * m

    time = @belapsed SymmetricEigenProblem.bulk_wave_order_2x2_rmul_parallel!($Q, $givens)

    return flops / time / 1e9
end