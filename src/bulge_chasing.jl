using LinearAlgebra: givensAlgorithm, SymTridiagonal, I, Diagonal
using Base: @propagate_inbounds

import LinearAlgebra: lmul!, rmul!
import Base: Matrix

@propagate_inbounds is_offdiagonal_small(H::SymTridiagonal{T}, i::Int, tol = eps(real(T))) where {T} = 
    abs(H.ev[i]) ≤ tol*(abs(H.dv[i]) + abs(H.dv[i+1]))

abstract type SmallRotation end

"""
Given's rotation acting on rows i:i+1
"""
struct Rotation2{Tc,Ts} <: SmallRotation
    c::Tc
    s::Ts
    i::Int
end

"""
Get a rotation that maps [p₁, p₂] to a multiple of [1, 0]
"""
function get_rotation(p₁, p₂, i::Int)
    c, s, nrm = givensAlgorithm(p₁, p₂)
    Rotation2(c, s, i), nrm
end

"""
Passed into the Schur factorization function if you do not wish to have the Schur vectors.
"""
struct NotWanted end

@inline lmul!(G::SmallRotation, A::AbstractMatrix) = lmul!(G, A, 1, size(A, 2))
@inline rmul!(A::AbstractMatrix, G::SmallRotation) = rmul!(A, G, 1, size(A, 1))

lmul!(::SmallRotation, ::NotWanted, args...) = nothing
rmul!(::NotWanted, ::SmallRotation, args...) = nothing

@inline function lmul!(G::Rotation2, A::AbstractMatrix, from::Int, to::Int)
    @inbounds @fastmath for j = from:to
        a₁ = A[G.i+0,j]
        a₂ = A[G.i+1,j]

        a₁′ = G.c * a₁ + G.s * a₂
        a₂′ = -G.s' * a₁ + G.c * a₂
        
        A[G.i+0,j] = a₁′
        A[G.i+1,j] = a₂′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation2, from::Int, to::Int)
    @inbounds @fastmath for j = from:to
        a₁ = A[j,G.i+0]
        a₂ = A[j,G.i+1]

        a₁′ = a₁ * G.c + a₂ * G.s'
        a₂′ = a₁ * -G.s + a₂ * G.c

        A[j,G.i+0] = a₁′
        A[j,G.i+1] = a₂′
    end
    A
end

function single_shift!(H::SymTridiagonal{Tv}, from::Int, to::Int, μ::Number, Q = NotWanted()) where {Tv<:Number}
    m, n = size(H)

    @inbounds @fastmath begin

        # Compute the nonzero entries of p = (H - μI)e₁.
        H₁₁ = H.dv[from]
        H₂₁ = H.ev[from]

        p₁ = H₁₁ - μ
        p₂ = H₂₁

        # Map that column to a multiple of e₁ via two Given's rotations
        G₁, nrm = get_rotation(p₁, p₂, from)

        # Apply the Given's rotations
        H₂₂ = H.dv[from+1]

        # First col
        H₁₁′ =  G₁.c * H₁₁ + G₁.s * H₂₁
        H₂₁′ = -G₁.s * H₁₁ + G₁.c * H₂₁

        # Second col
        H₁₂′ =  G₁.c * H₂₁ + G₁.s * H₂₂
        H₂₂′ = -G₁.s * H₂₁ + G₁.c * H₂₂

        # First row
        H₁₁′′ = H₁₁′ *  G₁.c + H₁₂′ * G₁.s # H₁₁ and H₁₂ touched by left mul
        H₁₂′′ = H₁₁′ * -G₁.s + H₁₂′ * G₁.c # H₁₁ and H₁₂ touched by left mul
        
        # Second row
        H₂₂′′ = H₂₁′ * -G₁.s + H₂₂′ * G₁.c # H₂₁ and H₂₂ touched by left mul

        # Store.
        H.dv[from + 0] = H₁₁′′
        H.dv[from + 1] = H₂₂′′
        H.ev[from + 0] = H₁₂′′

        # Conditonally store depending on whether the off-diagonal is still inbounds
        bulge_value = zero(Tv)
        
        if from + 2 ≤ to
            # Third col: the bulge-inducing element 
            H₂₃ = H.ev[from + 1]
            H₁₃′ = G₁.s * H₂₃
            H₂₃′ = G₁.c * H₂₃
            bulge_value = H₁₃′
            H.ev[from + 1] = H₂₃′
        end

        rmul!(Q, G₁)

        # Bulge chasing. First step of the for-loop below looks like:
        #  from           to
        #     ↓           ↓
        #     x x x             x x x             x + o        
        # i → x x x             + + + +           x + + x       
        #     x x x x           o + + +             + + x      
        #         x x x      ⇒      x x x      ⇒    + + x x    
        #       |   x x x             x x x             x x x  
        #       |     x x x             x x x             x x x
        #       |       x x               x x               x x
        #       ↑
        #       i
        #
        # Last iterations looks like:
        #  from           to
        #     ↓           ↓
        #     x x               x x               x x          
        #     x x x             x x x             x x x        
        #       x x x             x x x             x x x     
        #         x x x      ⇒      x x x      ⇒      x x x    
        #           x x x x           x x x             x x + o
        # i → ------- x x x             + + +             x + +
        #             x x x             0 + +               + +
        #               ↑
        #               i

        for i = from + 1 : to - 1
            
            p₁ = H.ev[i - 1]
            p₂ = bulge_value

            G, nrm = get_rotation(p₁, p₂, i)

            # Unbulging, we zero'd out the second off-diagonal
            # and we only store the off-diaognal
            H.ev[i - 1] = nrm

            # B for bulge. Load the relevant values
            B₁₁ = H.dv[i + 0]
            B₂₂ = H.dv[i + 1]
            B₂₁ = H.ev[i + 0]

            # First col
            B₁₁′ =  G.c * B₁₁ + G.s * B₂₁
            B₂₁′ = -G.s * B₁₁ + G.c * B₂₁

            # Second col
            B₁₂′ =  G.c * B₂₁ + G.s * B₂₂ # B₂₁ = B₁₂
            B₂₂′ = -G.s * B₂₁ + G.c * B₂₂ # B₂₁ = B₁₂

            # First row
            B₁₁′′ = B₁₁′ *  G.c + B₁₂′ * G.s # B₁₁ and B₁₂ touched by left mul
            B₁₂′′ = B₁₁′ * -G.s + B₁₂′ * G.c # B₁₁ and B₁₂ touched by left mul
            
            # Second row
            B₂₂′′ = B₂₁′ * -G.s + B₂₂′ * G.c # B₂₁ and B₂₂ touched by left mul

            # Store.
            H.dv[i + 0] = B₁₁′′
            H.dv[i + 1] = B₂₂′′
            H.ev[i + 0] = B₁₂′′

            # Conditonally store depending on whether the off-diagonal is still inbounds
            if i + 2 ≤ to
                # Third col: the bulge-inducing element
                B₂₃ = H.ev[i + 1]
                B₁₃′ = G.s * B₂₃
                B₂₃′ = G.c * B₂₃
                bulge_value = B₁₃′
                H.ev[i + 1] = B₂₃′
            end

            rmul!(Q, G)
        end
    end

    H
end

function qr_algorithm!(H::SymTridiagonal{T}, start::Int, to::Int, Q = NotWanted(), tol = eps(T), maxiter = 100*size(H, 1)) where {T<:Real}
    # iteration count
    iter = 0

    n = size(H, 1)

    @inbounds @fastmath while true

        if iter > maxiter
            return H, false, iter
        end

        # Indexing
        # `to` points to the column where the off-diagonal value was last zero.
        # while `from` points to the smallest index such that there is no small off-diagonal
        # value in columns from:end-1. Sometimes `from` is just 1. Cartoon of a split 
        # with from != 1:
        # 
        #  + +            
        #  + + o          
        #    o X X        
        #      X X X      
        #      . X X X    
        #      .   X X o 
        #      .     o + +
        #      .     . + +
        #      ^     ^
        #   from   to
        # The X's form the unreduced tridiagonal matrix we are applying QR iterations to,
        # the + values remain untouched! The o's are zeros -- or numerically considered zeros.

        # We keep `from` one column past the zero off-diagonal value, so we check whether
        # the `from - 1` column has a small off-diagonal value.
        from = to
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            H.ev[from-1] = zero(T)
            to -= 1
        else
            # Now we are sure we can work with a 2×2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁ = H.dv[to-1]
            H₂₂ = H.dv[to]
            H₁₂ = H.ev[to-1]

            # Scaling to avoid losing precision in the case where we have nearly
            # repeated eigenvalues.
            scale = abs(H₁₁) + 2abs(H₁₂) + abs(H₂₂)
            H₁₁ /= scale
            H₁₂ /= scale
            H₂₂ /= scale

            # Trace and discriminant of small eigenvalue problem.
            t = (H₁₁ + H₂₂) / 2
            d = (H₁₁ - t) * (H₂₂ - t) - H₁₂ * H₁₂
            sqrt_discr = sqrt(abs(d))

            # Real eigenvalues.
            # Note that if from + 1 == to in this case, then just one additional
            # iteration is necessary, since the Wilkinson shift will do an exact shift.

            # Determine the Wilkinson shift -- the closest eigenvalue of the 2x2 block
            # near H[to,to]
            
            λ₁ = t + sqrt_discr
            λ₂ = t - sqrt_discr
            λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
            λ *= scale

            # Run a bulge chase
            single_shift!(H, from, to, λ, Q)
            iter += 1
        end

        # Converged!
        to ≤ start && break
    end

    return H, true, iter
end

qr_algorithm!(H::SymTridiagonal{T}, Q = NotWanted(), tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T} = qr_algorithm!(H, 1, size(H, 2), Q, tol, maxiter)

function eigen_decomp!(H::SymTridiagonal{T}) where T
    n = size(H, 1)
    Q = Matrix{T}(I, n, n)

    _, converged, iter = qr_algorithm!(H, Q)

    @info "done" converged iter

    return Diagonal(H.dv), Q
end