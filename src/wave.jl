using Base.Threads: nthreads, @spawn, @sync

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

function bulk_wave_order_2x2_rmul_parallel!(A::AbstractMatrix, givens::Matrix{Tuple{T,T}}) where {T}
    p = Threads.nthreads()

    if p == 1
        bulk_wave_order_2x2_rmul_parallel!(A, givens)
        return nothing
    end

    # divide into p roughly equal parts where part % 4 == 0
    m = size(A, 1)
    part = 4 * (m ÷ p) ÷ 4

    @sync begin
        for i = 1:p
            @spawn begin
                from = 1 + part * (i - 1)
                to = i == p ? m : 1 + part * i - 1
                bulk_wave_order_2x2_rmul!(view(A, from:to, :), givens)
            end
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
