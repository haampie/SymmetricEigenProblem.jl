using Test
using SymmetricEigenProblem

@testset begin
    A = SymTridiagonal(rand(100), rand(99))

    B, converged = SymmetricEigenProblem.qr_algorithm!(copy(A))

    @test all(sort!(diag(B)) ≈ eigvals(A))
end