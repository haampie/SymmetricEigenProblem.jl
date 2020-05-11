using SIMDPirates
using BenchmarkTools
using Test
using LoopVectorization: @avx

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

        for row in axes(A, 1)
            A1 = A[row, 1]
            A2 = A[row, 2]
            A3 = A[row, 3]
            A4 = A[row, 4]

            # Apply rotation 1 to column 2 and 3
            A2′, A3′ = mul(G1, A2, A3)

            # Apply rotation 2 to column 3 and 4
            A3′′, A4′ = mul(G2, A3′, A4)

            # Apply rotation 3 to column 1 and 2
            A1′, A2′′ = mul(G3, A1, A2′)

            # Apply rotation 4 to column 2 and 3
            A2′′′, A3′′′ = mul(G4, A2′′, A3′′)

            A[row, 1] = A1′
            A[row, 2] = A2′′′
            A[row, 3] = A3′′′
            A[row, 4] = A4′
        end
    end

    return A
end

function loop_vec!(A::AbstractMatrix{Float64}, givens::AbstractArray{Float64})
    @inbounds begin
        c1 = givens[1]
        s1 = givens[2]
        c2 = givens[3]
        s2 = givens[4]
        c3 = givens[5]
        s3 = givens[6]
        c4 = givens[7]
        s4 = givens[8]

        @avx for row in axes(A, 1)
            A1 = A[row, 1]
            A2 = A[row, 2]
            A3 = A[row, 3]
            A4 = A[row, 4]

            # Apply rotation 1 to column 2 and 3
            A2′ =  c1 * A2 + s1 * A3
            A3′ = -s1 * A2 + c1 * A3

            # Apply rotation 2 to column 3 and 4
            A3′′ =  c2 * A3′ + s2 * A4
            A4′  = -s2 * A3′ + c2 * A4

            # Apply rotation 3 to column 1 and 2
            A1′  =  c3 * A1 + s3 * A2′
            A2′′ = -s3 * A1 + c3 * A2′

            # Apply rotation 4 to column 2 and 3
            A2′′′ =  c4 * A2′′ + s4 * A3′′
            A3′′′ = -s4 * A2′′ + c4 * A3′′

            A[row, 1] = A1′
            A[row, 2] = A2′′′
            A[row, 3] = A3′′′
            A[row, 4] = A4′
        end
    end

    return A
end

function avx_givens_first!(A::AbstractMatrix{Float64}, givens::AbstractArray{Float64})
    A_col_1 = pointer(A) + 0 * stride(A, 2) * sizeof(Float64)
    A_col_2 = pointer(A) + 1 * stride(A, 2) * sizeof(Float64)
    A_col_3 = pointer(A) + 2 * stride(A, 2) * sizeof(Float64)
    A_col_4 = pointer(A) + 3 * stride(A, 2) * sizeof(Float64)

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

        for i = Base.OneTo(size(A, 1) ÷ 4)
            # Load the columns
            col_1 = vload(SVec{4,Float64}, A_col_1)
            col_2 = vload(SVec{4,Float64}, A_col_2)
            col_3 = vload(SVec{4,Float64}, A_col_3)
            col_4 = vload(SVec{4,Float64}, A_col_4)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 2 to column 3 and 4
            col_3′′  = s2 * col_4 + c2 * col_3′
            col_4′   = c2 * col_4 - s2 * col_3′

            # Apply rotation 3 to column 1 and 2
            col_1′   = s3 * col_2′ + c3 * col_1
            col_2′′  = c3 * col_2′ - s3 * col_1

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′′ + c4 * col_2′′
            col_3′′′ = c4 * col_3′′ - s4 * col_2′′
            
            vstore!(A_col_1, col_1′  )
            vstore!(A_col_2, col_2′′′)
            vstore!(A_col_3, col_3′′′)
            vstore!(A_col_4, col_4′  )
            
            A_col_1 += 4 * sizeof(Float64)
            A_col_2 += 4 * sizeof(Float64)
            A_col_3 += 4 * sizeof(Float64)
            A_col_4 += 4 * sizeof(Float64)
        end

        return A
    end
end

function avx_givens_second!(A::AbstractMatrix{Float64}, givens::AbstractArray{Float64})
    A_col_1 = pointer(A) + 0 * stride(A, 2) * sizeof(Float64)
    A_col_2 = pointer(A) + 1 * stride(A, 2) * sizeof(Float64)
    A_col_3 = pointer(A) + 2 * stride(A, 2) * sizeof(Float64)
    A_col_4 = pointer(A) + 3 * stride(A, 2) * sizeof(Float64)

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

        for i = Base.OneTo(size(A, 1) ÷ 8)
            # Load the columns
            col_1 = vload(SVec{4,Float64}, A_col_1)
            col_5 = vload(SVec{4,Float64}, A_col_1 + 4 * sizeof(Float64))

            col_2 = vload(SVec{4,Float64}, A_col_2)
            col_6 = vload(SVec{4,Float64}, A_col_2 + 4 * sizeof(Float64))

            col_3 = vload(SVec{4,Float64}, A_col_3)
            col_7 = vload(SVec{4,Float64}, A_col_3 + 4 * sizeof(Float64))

            col_4 = vload(SVec{4,Float64}, A_col_4)
            col_8 = vload(SVec{4,Float64}, A_col_4 + 4 * sizeof(Float64))
            
            # Appy the Z-shape rotations

            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2
            col_6′   = s1 * col_7 + c1 * col_6
            col_7′   = c1 * col_7 - s1 * col_6

            # Apply rotation 2 to column 3 and 4
            col_3′′  = s2 * col_4 + c2 * col_3′
            col_4′   = c2 * col_4 - s2 * col_3′
            col_7′′  = s2 * col_8 + c2 * col_7′
            col_8′   = c2 * col_8 - s2 * col_7′

            # Apply rotation 3 to column 1 and 2
            col_1′   = s3 * col_2′ + c3 * col_1
            col_2′′  = c3 * col_2′ - s3 * col_1
            col_5′   = s3 * col_6′ + c3 * col_5
            col_6′′  = c3 * col_6′ - s3 * col_5

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′′ + c4 * col_2′′
            col_3′′′ = c4 * col_3′′ - s4 * col_2′′
            col_6′′′ = s4 * col_7′′ + c4 * col_6′′
            col_7′′′ = c4 * col_7′′ - s4 * col_6′′
            
            vstore!(A_col_1, col_1′  )
            vstore!(A_col_1 + 4 * sizeof(Float64), col_5′  )

            vstore!(A_col_2, col_2′′′)
            vstore!(A_col_2 + 4 * sizeof(Float64), col_6′′′)

            vstore!(A_col_3, col_3′′′)
            vstore!(A_col_3 + 4 * sizeof(Float64), col_7′′′)

            vstore!(A_col_4, col_4′  )
            vstore!(A_col_4 + 4 * sizeof(Float64), col_8′  )
            
            A_col_1 += 8 * sizeof(Float64)
            A_col_2 += 8 * sizeof(Float64)
            A_col_3 += 8 * sizeof(Float64)
            A_col_4 += 8 * sizeof(Float64)
        end

        return A
    end
end

function bench(n = 8 * 2000, w = 4)
    A = rand(Float64, n, w)
    givens = rand(Float64, 8)

    # 4 rotations, 4 muls + 2 adds per rotation, n rows
    flops = 4 * 6 * n

    ref = @belapsed reference_impl!($A, $givens)
    avx_first = @belapsed avx_givens_first!($A, $givens)
    avx_second = @belapsed avx_givens_second!($A, $givens)
    loop_vec = @belapsed loop_vec!($A, $givens)

    flops / ref / 1e9, flops / avx_first / 1e9, flops / avx_second / 1e9, flops / loop_vec / 1e9
end

function test()
    A = rand(Float64, 8 * 2000, 4)
    givens = rand(Float64, 8)

    A′ = reference_impl!(copy(A), givens)
    B′ = avx_givens_first!(copy(A), givens)
    C′ = avx_givens_second!(copy(A), givens)
    D′ = loop_vec!(copy(A), givens)

    @test(A′ ≈ B′), @test(A′ ≈ C′), @test(A′ ≈ D′)
end