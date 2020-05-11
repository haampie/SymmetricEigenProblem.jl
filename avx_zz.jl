using SIMDPirates
using BenchmarkTools
using Test

struct Rot{T}
    c::T
    s::T
end

mul(G::Rot, a, b) = (G.c * a + G.s * b, -G.s * a + G.c * b)

function reference_impl!(A::AbstractMatrix{Float64}, givens::AbstractArray{Float64})
    @inbounds begin
        G1 = Rot(givens[1], givens[2])
        G2 = Rot(givens[3], givens[4])
        G3 = Rot(givens[5], givens[6])
        G4 = Rot(givens[7], givens[8])
        G5 = Rot(givens[9], givens[10])
        G6 = Rot(givens[11], givens[12])

        for row in axes(A, 1)
            A1 = A[row, 1]
            A2 = A[row, 2]
            A3 = A[row, 3]
            A4 = A[row, 4]
            A5 = A[row, 5]

            # Apply rotation 1 to column 3 and 4
            A3′, A4′ = mul(G1, A3, A4)

            # Apply rotation 2 to column 4 and 5
            A4′′, A5′ = mul(G2, A4′, A5)

            # Apply rotation 3 to column 2 and 3
            A2′, A3′′ = mul(G3, A2, A3′)

            # Apply rotation 4 to column 3 and 4
            A3′′′, A4′′′ = mul(G4, A3′′, A4′′)

            # Apply rotation 5 to column 1 and 2
            A1′, A2′′ = mul(G5, A1, A2′)

            # Apply rotation 5 to column 2 and 3
            A2′′′, A3′′′′ = mul(G6, A2′′, A3′′′)

            A[row, 1] = A1′
            A[row, 2] = A2′′′
            A[row, 3] = A3′′′′
            A[row, 4] = A4′′′
            A[row, 5] = A5′
        end
    end

    return A
end

function avx_zz!(A::AbstractMatrix{Float64}, givens::AbstractArray{Float64})
    A_col_1 = pointer(A) + 0 * stride(A, 2) * sizeof(Float64)
    A_col_2 = pointer(A) + 1 * stride(A, 2) * sizeof(Float64)
    A_col_3 = pointer(A) + 2 * stride(A, 2) * sizeof(Float64)
    A_col_4 = pointer(A) + 3 * stride(A, 2) * sizeof(Float64)
    A_col_5 = pointer(A) + 4 * stride(A, 2) * sizeof(Float64)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(SVec{4,Float64}, givens[1])
        s1 = vbroadcast(SVec{4,Float64}, givens[2])
        c2 = vbroadcast(SVec{4,Float64}, givens[3])
        s2 = vbroadcast(SVec{4,Float64}, givens[4])
        c3 = vbroadcast(SVec{4,Float64}, givens[5])
        s3 = vbroadcast(SVec{4,Float64}, givens[6])
        c4 = vbroadcast(SVec{4,Float64}, givens[7])
        s4 = vbroadcast(SVec{4,Float64}, givens[8])
        c5 = vbroadcast(SVec{4,Float64}, givens[9])
        s5 = vbroadcast(SVec{4,Float64}, givens[10])
        c6 = vbroadcast(SVec{4,Float64}, givens[11])
        s6 = vbroadcast(SVec{4,Float64}, givens[12])

        for i = Base.OneTo(size(A, 1) ÷ 4)
            # Load the columns
            col_1 = vload(SVec{4,Float64}, A_col_1)
            col_2 = vload(SVec{4,Float64}, A_col_2)
            col_3 = vload(SVec{4,Float64}, A_col_3)
            col_4 = vload(SVec{4,Float64}, A_col_4)
            col_5 = vload(SVec{4,Float64}, A_col_5)
            
            # Apply rotation 1 to column 3 and 4
            col_3′    = s1 * col_4 + c1 * col_3
            col_4′    = c1 * col_4 - s1 * col_3

            # Apply rotation 2 to column 4 and 5
            col_4′′   = s2 * col_5 + c2 * col_4′
            col_5′    = c2 * col_5 - s2 * col_4′

            # Apply rotation 3 to column 2 and 3
            col_2′    = s3 * col_3′ + c3 * col_2
            col_3′′   = c3 * col_3′ - s3 * col_2

            # Apply rotation 4 to column 3 and 4
            col_3′′′  = s4 * col_4′′ + c4 * col_3′′
            col_4′′′  = c4 * col_4′′ - s4 * col_3′′

            # Apply rotation 5 to column 1 and 2
            col_1′  = s5 * col_2′ + c5 * col_1
            col_2′′ = c5 * col_2′ - s5 * col_1
            
            # Apply rotation 6 to column 2 and 3
            col_2′′′  = s6 * col_3′′′ + c6 * col_2′′
            col_3′′′′ = c6 * col_3′′′ - s6 * col_2′′
            
            vstore!(A_col_1, col_1′   )
            vstore!(A_col_2, col_2′′′ )
            vstore!(A_col_3, col_3′′′′)
            vstore!(A_col_4, col_4′′′ )
            vstore!(A_col_5, col_5′ )
            
            A_col_1 += 4 * sizeof(Float64)
            A_col_2 += 4 * sizeof(Float64)
            A_col_3 += 4 * sizeof(Float64)
            A_col_4 += 4 * sizeof(Float64)
            A_col_5 += 4 * sizeof(Float64)
        end

        return A
    end
end

function bench_zz(n = 8 * 2000, w = 5)
    A = rand(Float64, n, w)
    givens = rand(Float64, 12)

    # 6 rotations, 4 muls + 2 adds per rotation, n rows
    flops = 6 * 6 * n

    ref = @belapsed reference_impl!($A, $givens)
    avx = @belapsed avx_zz!($A, $givens)

    flops / ref / 1e9, flops / avx / 1e9
end

function test_zz()
    A = rand(Float64, 8 * 2000, 5)
    givens = rand(Float64, 12)

    A′ = reference_impl!(copy(A), givens)
    B′ = avx_zz!(copy(A), givens)

    @test(A′ ≈ B′)
end