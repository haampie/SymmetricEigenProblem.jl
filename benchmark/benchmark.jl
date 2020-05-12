using SymmetricEigenProblem
using BenchmarkTools

function lapack()
    A = 
    LAPACK.syev!
end

function bench(m = 2000, n = 2000, k = 64)

    givens = fill((rand(), rand()), n - 1, k)

    Q = rand(m, n)

    # 6 flops per givens rotation per row
    flops = 6 * k * (n - 1) * m

    time = @belapsed SymmetricEigenProblem.bulk_wave_order_2x2_rmul!($Q, $givens)

    return flops / time / 1e9
end