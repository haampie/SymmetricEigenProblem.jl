using SIMDPirates
using VectorizationBase: pick_vector_width, pick_vector_width_shift, mask

function givens_kernel!(A::AbstractMatrix{T}, givens::Fused2x2{T}) where T
    m = size(A, 1)

    # Find register size, number of loops, size of remainder
    W, Wshift = pick_vector_width_shift(T)
    V = SVec{W,T}
    num_full_loops = m >> Wshift
    
    # Â points to the first entry in the first column
    # we apply rotations to
    Â = pointer(A) + (givens.i - 1) * stride(A, 2) * sizeof(T)
    
    # From Â we can access the other 4 columns
    # These offsets are computed here explicitly such that they are stored in
    # registers. When using 4 pointers, the compiler is not always smart enough
    # to figure out it only has to increment 1 pointer and use relative offsets
    offset_col_1 = 0 * stride(A, 2) * sizeof(T)
    offset_col_2 = 1 * stride(A, 2) * sizeof(T)
    offset_col_3 = 2 * stride(A, 2) * sizeof(T)
    offset_col_4 = 3 * stride(A, 2) * sizeof(T)

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
            col_1 = vload(V, Â + offset_col_1)
            col_2 = vload(V, Â + offset_col_2)
            col_3 = vload(V, Â + offset_col_3)
            col_4 = vload(V, Â + offset_col_4)
            
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
            
            vstore!(Â + offset_col_1, col_1′  )
            vstore!(Â + offset_col_2, col_2′′′)
            vstore!(Â + offset_col_3, col_3′′′)
            vstore!(Â + offset_col_4, col_4′  )
            
            Â += W * sizeof(T)
        end

        #Stuff that does not fit in the register, used masks
        let
            remainder = m & (W - 1)
            remainder == 0 && return nothing
            remainder_mask = mask(T, remainder)
            
            # Load the columns
            col_1 = vload(V, Â + offset_col_1, remainder_mask)
            col_2 = vload(V, Â + offset_col_2, remainder_mask)
            col_3 = vload(V, Â + offset_col_3, remainder_mask)
            col_4 = vload(V, Â + offset_col_4, remainder_mask)
            
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
            
            vstore!(Â + offset_col_1, col_1′  , remainder_mask)
            vstore!(Â + offset_col_2, col_2′′′, remainder_mask)
            vstore!(Â + offset_col_3, col_3′′′, remainder_mask)
            vstore!(Â + offset_col_4, col_4′  , remainder_mask)
        end
    end
    
    return nothing
end

function givens_kernel!(A::AbstractMatrix{T}, givens::Fused2x2_remainder1{T}) where T
    m = size(A, 1)

    # Find register size, number of loops, size of remainder
    W, Wshift = pick_vector_width_shift(T)
    V = SVec{W,T}
    num_full_loops = m >> Wshift
    
    # Â points to the first entry in the first column
    # we apply rotations to
    Â = pointer(A) + (givens.i - 1) * stride(A, 2) * sizeof(T)
    
    # From Â we can access the other 4 columns
    # These offsets are computed here explicitly such that they are stored in
    # registers. When using 4 pointers, the compiler is not always smart enough
    # to figure out it only has to increment 1 pointer and use relative offsets
    offset_col_1 = 0 * stride(A, 2) * sizeof(T)
    offset_col_2 = 1 * stride(A, 2) * sizeof(T)
    offset_col_3 = 2 * stride(A, 2) * sizeof(T)
    offset_col_4 = 3 * stride(A, 2) * sizeof(T)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(V, givens.c1)
        s1 = vbroadcast(V, givens.s1)
        c2 = vbroadcast(V, givens.c2)
        s2 = vbroadcast(V, givens.s2)
        c4 = vbroadcast(V, givens.c4)
        s4 = vbroadcast(V, givens.s4)

        for i = Base.OneTo(num_full_loops)
            # Load the columns
            col_2 = vload(V, Â + offset_col_2)
            col_3 = vload(V, Â + offset_col_3)
            col_4 = vload(V, Â + offset_col_4)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 2 to column 3 and 4
            col_3′′  = s2 * col_4 + c2 * col_3′
            col_4′   = c2 * col_4 - s2 * col_3′

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′′ + c4 * col_2′
            col_3′′′ = c4 * col_3′′ - s4 * col_2′
            
            vstore!(Â + offset_col_2, col_2′′′)
            vstore!(Â + offset_col_3, col_3′′′)
            vstore!(Â + offset_col_4, col_4′  )
            
            Â += W * sizeof(T)
        end

        #Stuff that does not fit in the register, used masks
        let
            remainder = m & (W - 1)
            remainder == 0 && return nothing
            remainder_mask = mask(T, remainder)
            
            # Load the columns
            col_2 = vload(V, Â + offset_col_2, remainder_mask)
            col_3 = vload(V, Â + offset_col_3, remainder_mask)
            col_4 = vload(V, Â + offset_col_4, remainder_mask)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 2 to column 3 and 4
            col_3′′  = s2 * col_4 + c2 * col_3′
            col_4′   = c2 * col_4 - s2 * col_3′

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′′ + c4 * col_2′
            col_3′′′ = c4 * col_3′′ - s4 * col_2′

            vstore!(Â + offset_col_2, col_2′′′, remainder_mask)
            vstore!(Â + offset_col_3, col_3′′′, remainder_mask)
            vstore!(Â + offset_col_4, col_4′  , remainder_mask)
        end
    end
    
    return nothing
end

function givens_kernel!(A::AbstractMatrix{T}, givens::Fused2x2_remainder2{T}) where T
    m = size(A, 1)

    # Find register size, number of loops, size of remainder
    W, Wshift = pick_vector_width_shift(T)
    V = SVec{W,T}
    num_full_loops = m >> Wshift
    
    # Â points to the first entry in the first column
    # we apply rotations to
    Â = pointer(A) + (givens.i - 1) * stride(A, 2) * sizeof(T)
    
    # From Â we can access the other 4 columns
    # These offsets are computed here explicitly such that they are stored in
    # registers. When using 4 pointers, the compiler is not always smart enough
    # to figure out it only has to increment 1 pointer and use relative offsets
    offset_col_1 = 0 * stride(A, 2) * sizeof(T)
    offset_col_2 = 1 * stride(A, 2) * sizeof(T)
    offset_col_3 = 2 * stride(A, 2) * sizeof(T)
    offset_col_4 = 3 * stride(A, 2) * sizeof(T)

    @inbounds begin
        # Load the rotations
        c1 = vbroadcast(V, givens.c1)
        s1 = vbroadcast(V, givens.s1)
        c3 = vbroadcast(V, givens.c3)
        s3 = vbroadcast(V, givens.s3)
        c4 = vbroadcast(V, givens.c4)
        s4 = vbroadcast(V, givens.s4)

        for i = Base.OneTo(num_full_loops)
            # Load the columns
            col_1 = vload(V, Â + offset_col_1)
            col_2 = vload(V, Â + offset_col_2)
            col_3 = vload(V, Â + offset_col_3)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 3 to column 1 and 2
            col_1′   = s3 * col_2′ + c3 * col_1
            col_2′′  = c3 * col_2′ - s3 * col_1

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′ + c4 * col_2′′
            col_3′′′ = c4 * col_3′ - s4 * col_2′′
            
            vstore!(Â + offset_col_1, col_1′  )
            vstore!(Â + offset_col_2, col_2′′′)
            vstore!(Â + offset_col_3, col_3′′′)
            
            Â += W * sizeof(T)
        end

        #Stuff that does not fit in the register, used masks
        let
            remainder = m & (W - 1)
            
            remainder == 0 && return nothing

            remainder_mask = mask(T, remainder)
            
            # Load the columns
            col_1 = vload(V, Â + offset_col_1, remainder_mask)
            col_2 = vload(V, Â + offset_col_2, remainder_mask)
            col_3 = vload(V, Â + offset_col_3, remainder_mask)
            
            # Apply rotation 1 to column 2 and 3
            col_2′   = s1 * col_3 + c1 * col_2
            col_3′   = c1 * col_3 - s1 * col_2

            # Apply rotation 3 to column 1 and 2
            col_1′   = s3 * col_2′ + c3 * col_1
            col_2′′  = c3 * col_2′ - s3 * col_1

            # Apply rotation 4 to column 2 and 3
            col_2′′′ = s4 * col_3′ + c4 * col_2′′
            col_3′′′ = c4 * col_3′ - s4 * col_2′′
            
            vstore!(Â + offset_col_1, col_1′  , remainder_mask)
            vstore!(Â + offset_col_2, col_2′′′, remainder_mask)
            vstore!(Â + offset_col_3, col_3′′′, remainder_mask)
        end
    end
    
    return nothing
end