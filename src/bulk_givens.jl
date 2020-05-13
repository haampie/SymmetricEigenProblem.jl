using SIMDPirates
using VectorizationBase: pick_vector_width, pick_vector_width_shift, mask

function givens_kernel!(A::AbstractMatrix{T}, givens::Fused2x2{T}) where T
    m = size(A, 1)

    # Find register size, number of loops, size of remainder
    N = pick_vector_width(T)
    W, Wshift = pick_vector_width_shift(T)
    V = SVec{N,T}
    num_full_loops = m >> Wshift
    remainder = m & (W - 1)
    remainder_mask = mask(Float64, 5)
    
    # Set up some pointers to the columns of A
    A_col_1 = pointer(A) + (0 + givens.i - 1) * stride(A, 2) * sizeof(T)
    A_col_2 = pointer(A) + (1 + givens.i - 1) * stride(A, 2) * sizeof(T)
    A_col_3 = pointer(A) + (2 + givens.i - 1) * stride(A, 2) * sizeof(T)
    A_col_4 = pointer(A) + (3 + givens.i - 1) * stride(A, 2) * sizeof(T)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(V, givens.c1)
        s1 = vbroadcast(V, givens.s1)
        c2 = vbroadcast(V, givens.c2)
        s2 = vbroadcast(V, givens.s2)
        c3 = vbroadcast(V, givens.c3)
        s3 = vbroadcast(V, givens.s3)
        c4 = vbroadcast(V, givens.c4)
        s4 = vbroadcast(V, givens.s4)

        for i = Base.OneTo(num_full_loops)
            # Load the columns
            col_1 = vload(V, A_col_1)
            col_2 = vload(V, A_col_2)
            col_3 = vload(V, A_col_3)
            col_4 = vload(V, A_col_4)
            
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
            
            A_col_1 += N * sizeof(Float64)
            A_col_2 += N * sizeof(Float64)
            A_col_3 += N * sizeof(Float64)
            A_col_4 += N * sizeof(Float64)
        end

        # Stuff that does not fit in the register, used masks
        let
            # Load the columns
            col_1 = vload(V, A_col_1, remainder_mask)
            col_2 = vload(V, A_col_2, remainder_mask)
            col_3 = vload(V, A_col_3, remainder_mask)
            col_4 = vload(V, A_col_4, remainder_mask)
            
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
            
            vstore!(A_col_1, col_1′  , remainder_mask)
            vstore!(A_col_2, col_2′′′, remainder_mask)
            vstore!(A_col_3, col_3′′′, remainder_mask)
            vstore!(A_col_4, col_4′  , remainder_mask)
        end
    end
    
    return nothing
end

function givens_kernel!(A::AbstractMatrix{Float64}, givens::Fused2x2_remainder1)
    #A_col_1 = pointer(A) + (0 + givens.i - 1) * stride(A, 2) * sizeof(Float64)
    A_col_2 = pointer(A) + (1 + givens.i - 1) * stride(A, 2) * sizeof(Float64)
    A_col_3 = pointer(A) + (2 + givens.i - 1) * stride(A, 2) * sizeof(Float64)
    A_col_4 = pointer(A) + (3 + givens.i - 1) * stride(A, 2) * sizeof(Float64)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(SVec{4,Float64}, givens.c1)
        s1 = vbroadcast(SVec{4,Float64}, givens.s1)
        c2 = vbroadcast(SVec{4,Float64}, givens.c2)
        s2 = vbroadcast(SVec{4,Float64}, givens.s2)
        c4 = vbroadcast(SVec{4,Float64}, givens.c4)
        s4 = vbroadcast(SVec{4,Float64}, givens.s4)

        for i = Base.OneTo(size(A, 1) ÷ 4)
            # Load the columns
            col_2 = vload(SVec{4,Float64}, A_col_2)
            col_3 = vload(SVec{4,Float64}, A_col_3)
            col_4 = vload(SVec{4,Float64}, A_col_4)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 2 to column 3 and 4
            col_3′′  = s2 * col_4 + c2 * col_3′
            col_4′   = c2 * col_4 - s2 * col_3′

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′′ + c4 * col_2′
            col_3′′′ = c4 * col_3′′ - s4 * col_2′

            vstore!(A_col_2, col_2′′′)
            vstore!(A_col_3, col_3′′′)
            vstore!(A_col_4, col_4′  )

            A_col_2 += 4 * sizeof(Float64)
            A_col_3 += 4 * sizeof(Float64)
            A_col_4 += 4 * sizeof(Float64)
        end
    end
    
    return nothing
end

function givens_kernel!(A::AbstractMatrix{Float64}, givens::Fused2x2_remainder2)
    A_col_1 = pointer(A) + (0 + givens.i - 1) * stride(A, 2) * sizeof(Float64)
    A_col_2 = pointer(A) + (1 + givens.i - 1) * stride(A, 2) * sizeof(Float64)
    A_col_3 = pointer(A) + (2 + givens.i - 1) * stride(A, 2) * sizeof(Float64)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(SVec{4,Float64}, givens.c1)
        s1 = vbroadcast(SVec{4,Float64}, givens.s1)
        c3 = vbroadcast(SVec{4,Float64}, givens.c3)
        s3 = vbroadcast(SVec{4,Float64}, givens.s3)
        c4 = vbroadcast(SVec{4,Float64}, givens.c4)
        s4 = vbroadcast(SVec{4,Float64}, givens.s4)

        for i = Base.OneTo(size(A, 1) ÷ 4)
            # Load the columns
            col_1 = vload(SVec{4,Float64}, A_col_1)
            col_2 = vload(SVec{4,Float64}, A_col_2)
            col_3 = vload(SVec{4,Float64}, A_col_3)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 3 to column 1 and 2
            col_1′   = s3 * col_2′ + c3 * col_1
            col_2′′  = c3 * col_2′ - s3 * col_1

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′ + c4 * col_2′′
            col_3′′′ = c4 * col_3′ - s4 * col_2′′
            
            vstore!(A_col_1, col_1′  )
            vstore!(A_col_2, col_2′′′)
            vstore!(A_col_3, col_3′′′)
            
            A_col_1 += 4 * sizeof(Float64)
            A_col_2 += 4 * sizeof(Float64)
            A_col_3 += 4 * sizeof(Float64)
        end
    end
    
    return nothing
end