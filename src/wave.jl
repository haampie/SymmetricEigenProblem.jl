struct Fused2x2{T}
    c1::T
    s1::T
    c2::T
    s2::T
    c3::T
    s3::T
    c4::T
    s4::T
    i::Int
end

struct Fused2x2_remainder1{T}
    c1::T
    s1::T
    c2::T
    s2::T
    c4::T
    s4::T
    i::Int
end

struct Fused2x2_remainder2{T}
    c1::T
    s1::T
    c3::T
    s3::T
    c4::T
    s4::T
    i::Int
end

# reference
function bulk_wave_order_2x2_rmul_!(A::AbstractMatrix, givens::Matrix{Tuple{T,T}}) where {T}
    for layer = axes(givens, 2)
        for col = axes(givens, 1)
            c, s = givens[col, layer]

            rmul!(A, Rotation2(c, s, col))
        end
    end

    return nothing
end

function bulk_wave_order_2x2_rmul!(A::AbstractMatrix, givens::Matrix{Tuple{T,T}}) where {T}
    m, n = size(givens)

    @assert rem(m, 2) == 1
    @assert rem(n, 2) == 0

    max_m = 2 * (m ÷ 2)

    @inbounds begin

        # Fan in
        for i = 2:2:n
            for j = 1:2:i-2
                start_col = i - j - 1

                c1, s1 = givens[start_col + 1, j + 0]
                c2, s2 = givens[start_col + 2, j + 0]
                c3, s3 = givens[start_col + 0, j + 1]
                c4, s4 = givens[start_col + 1, j + 1]

                # Skip identity rotations
                (s1 == s2 == s3 == s4 == zero(T)) && continue

                fused = Fused2x2(c1, s1, c2, s2, c3, s3, c4, s4, start_col)

                givens_kernel!(A, fused)
            end

            # Fix up the last guy
            let start_col = 0
                c1, s1 = givens[1, i - 1]
                c2, s2 = givens[2, i - 1]
                #c3, s3 = cut off
                c4, s4 = givens[1, i - 0]

                # Skip identity rotations
                if (s1 == s2 == s4 == zero(T)) == false
                    fused = Fused2x2_remainder1(c1, s1, c2, s2, c4, s4, 0)
                    givens_kernel!(A, fused)
                end
            end        
        end
        
        # Fast zone
        for i = n+2:2:max_m
            for j = 1:2:n
                start_col = i - j - 1

                c1, s1 = givens[start_col + 1, j + 0]
                c2, s2 = givens[start_col + 2, j + 0]
                c3, s3 = givens[start_col + 0, j + 1]
                c4, s4 = givens[start_col + 1, j + 1]

                # Skip identity rotations
                s1 == s2 == s3 == s4 == zero(T) && continue

                fused = Fused2x2(c1, s1, c2, s2, c3, s3, c4, s4, start_col)

                givens_kernel!(A, fused)
            end
        end

        # Fan out
        for i = max_m+2:2:m+n
            
            # Fix up the first guy
            let start_col = max_m
                layer = i - 1 - max_m

                c1, s1 = givens[start_col + 1, layer + 0]
                c3, s3 = givens[start_col + 0, layer + 1]
                c4, s4 = givens[start_col + 1, layer + 1]

                # Skip identity rotations
                if (s1 == s3 == s4 == zero(T)) == false
                    fused = Fused2x2_remainder2(c1, s1, c3, s3, c4, s4, start_col)
                    givens_kernel!(A, fused)
                end
            end

            for j = (i + 1 - max_m):2:n
                start_col = i - j - 1

                c1, s1 = givens[start_col + 1, j + 0]
                c2, s2 = givens[start_col + 2, j + 0]
                c3, s3 = givens[start_col + 0, j + 1]
                c4, s4 = givens[start_col + 1, j + 1]

                # Skip identity rotations
                s1 == s2 == s3 == s4 == zero(T) && continue

                fused = Fused2x2(c1, s1, c2, s2, c3, s3, c4, s4, start_col)

                givens_kernel!(A, fused)
            end
        end
    end

    nothing
end

function example(givens::Matrix{Tuple{T,T}}) where {T}
  
    m, n = size(givens)

    output = zeros(Int, m, n)

    @assert rem(m, 2) == 1
    @assert rem(n, 2) == 0

    max_m = 2 * (m ÷ 2)

    # Fan in
    for i = 2:2:n
        for j = 1:2:i-2
            start_col = i - j - 1

            output[start_col + 1, j + 0] = i
            output[start_col + 2, j + 0] = i
            output[start_col + 0, j + 1] = i
            output[start_col + 1, j + 1] = i
        end

        # Fix up the last guy
        let start_col = 0
            output[1, i - 0] = i
            output[1, i - 1] = i
            #c3, s3 = cut off
            output[2, i - 1] = i
        end        
    end
    
    # Fast zone
    for i = n+2:2:max_m
        for j = 1:2:n
            start_col = i - j - 1

            output[start_col + 1, j + 0] = i
            output[start_col + 2, j + 0] = i
            output[start_col + 0, j + 1] = i
            output[start_col + 1, j + 1] = i
        end
    end

    # Fan out
    for i = max_m+2:2:m+n
        
        # Fix up the first guy
        let start_col = max_m
            layer = i - 1 - max_m

            output[start_col + 1, layer + 0] = i
            output[start_col + 0, layer + 1] = i
            output[start_col + 1, layer + 1] = i
        end

        for j = (i + 1 - max_m):2:n
            start_col = i - j - 1

            output[start_col + 1, j + 0] = i
            output[start_col + 2, j + 0] = i
            output[start_col + 0, j + 1] = i
            output[start_col + 1, j + 1] = i
        end
    end

    output
end
