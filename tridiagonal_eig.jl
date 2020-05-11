using LinearAlgebra: givensAlgorithm, SymTridiagonal
using Base: @propagate_inbounds

import LinearAlgebra: lmul!, rmul!
import Base: Matrix

@propagate_inbounds is_offdiagonal_small(H::AbstractMatrix{T}, i::Int, tol = eps(real(T))) where {T} = 
    abs(H[i+1,i]) ≤ tol*(abs(H[i,i]) + abs(H[i+1,i+1]))

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
Two Given's rotations acting on rows i:i+2. This could also be implemented as one Householder
reflector!
"""
struct Rotation3{Tc,Ts} <: SmallRotation
    c₁::Tc
    s₁::Ts
    c₂::Tc
    s₂::Ts
    i::Int
end

# Some utility to materialize a rotation to a matrix.
function Matrix(r::Rotation2{Tc,Ts}, n::Int) where {Tc,Ts}
    r.i < n || throw(ArgumentError("Matrix should have order $(r.i+1) or larger"))
    G = Matrix{promote_type(Tc,Ts)}(I, n, n)
    G[r.i+0,r.i+0] = r.c
    G[r.i+1,r.i+0] = -conj(r.s)
    G[r.i+0,r.i+1] = r.s
    G[r.i+1,r.i+1] = r.c
    return G
end

function Matrix(r::Rotation3{Tc,Ts}, n::Int) where {Tc,Ts}
    G₁ = Matrix(Rotation2(r.c₁, r.s₁, r.i + 1), n)
    G₂ = Matrix(Rotation2(r.c₂, r.s₂, r.i), n)
    return G₂ * G₁
end

"""
Get a rotation that maps [p₁, p₂] to a multiple of [1, 0]
"""
function get_rotation(p₁, p₂, i::Int)
    c, s, nrm = givensAlgorithm(p₁, p₂)
    Rotation2(c, s, i), nrm
end

"""
Get a rotation that maps [p₁, p₂, p₃] to a multiple of [1, 0, 0]
"""
function get_rotation(p₁, p₂, p₃, i::Int)
    c₁, s₁, nrm₁ = givensAlgorithm(p₂, p₃)
    c₂, s₂, nrm₂ = givensAlgorithm(p₁, nrm₁)
    Rotation3(c₁, s₁, c₂, s₂, i), nrm₂
end

"""
Passed into the Schur factorization function if you do not wish to have the Schur vectors.
"""
struct NotWanted end

@inline lmul!(G::SmallRotation, A::AbstractMatrix) = lmul!(G, A, 1, size(A, 2))
@inline rmul!(A::AbstractMatrix, G::SmallRotation) = rmul!(A, G, 1, size(A, 1))

lmul!(::SmallRotation, ::NotWanted, args...) = nothing
rmul!(::NotWanted, ::SmallRotation, args...) = nothing

@inline function lmul!(G::Rotation3, A::AbstractMatrix, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[G.i+0,j]
        a₂ = A[G.i+1,j]
        a₃ = A[G.i+2,j]

        a₂′ = G.c₁ * a₂ + G.s₁ * a₃
        a₃′ = -G.s₁' * a₂ + G.c₁ * a₃

        a₁′′ = G.c₂ * a₁ + G.s₂ * a₂′
        a₂′′ = -G.s₂' * a₁ + G.c₂ * a₂′
        
        A[G.i+0,j] = a₁′′
        A[G.i+1,j] = a₂′′
        A[G.i+2,j] = a₃′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation3, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[j,G.i+0]
        a₂ = A[j,G.i+1]
        a₃ = A[j,G.i+2]

        a₂′ = a₂ * G.c₁ + a₃ * G.s₁'
        a₃′ = a₂ * -G.s₁ + a₃ * G.c₁

        a₁′′ = a₁ * G.c₂ + a₂′ * G.s₂'
        a₂′′ = a₁ * -G.s₂ + a₂′ * G.c₂

        A[j,G.i+0] = a₁′′
        A[j,G.i+1] = a₂′′
        A[j,G.i+2] = a₃′
    end
    A
end

@inline function lmul!(G::Rotation2, A::AbstractMatrix, from::Int, to::Int)
    @inbounds for j = from:to
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
    @inbounds for j = from:to
        a₁ = A[j,G.i+0]
        a₂ = A[j,G.i+1]

        a₁′ = a₁ * G.c + a₂ * G.s'
        a₂′ = a₁ * -G.s + a₂ * G.c

        A[j,G.i+0] = a₁′
        A[j,G.i+1] = a₂′
    end
    A
end

function single_shift_schur_tri!(H::AbstractMatrix{Tv}, from::Int, to::Int, μ::Number, Q = NotWanted()) where {Tv<:Number}
    m, n = size(H)

    # Compute the nonzero entries of p = (H - μI)e₁.
    @inbounds H₁₁ = H[from+0,from+0]
    @inbounds H₂₁ = H[from+1,from+0]

    p₁ = H₁₁ - μ
    p₂ = H₂₁

    # Map that column to a multiple of e₁ via two Given's rotations
    G₁, nrm = get_rotation(p₁, p₂, from)

    # Apply the Given's rotations
    lmul!(G₁, H, from, min(from + 2, n))
    rmul!(H, G₁, from, min(from + 2, m))
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

    @inbounds for i = from + 1 : to - 1
        
        p₁ = H[i + 0, i - 1]
        p₂ = H[i + 1, i - 1]

        G, nrm = get_rotation(p₁, p₂, i)

        # First column is done by hand
        H[i + 0, i - 1] = nrm
        H[i + 1, i - 1] = zero(Tv)
        
        # Rotate remaining columns
        lmul!(G, H, i, min(i + 2, n))

        # First row is done by hand
        H[i - 1, i + 0] = nrm
        H[i - 1, i + 1] = zero(Tv)

        # Create a new bulge
        rmul!(H, G, i, min(i + 2, m))
        rmul!(Q, G)
    end

    H
end

function single_shift_schur_tri!(H::SymTridiagonal{Tv}, from::Int, to::Int, μ::Number, Q = NotWanted()) where {Tv<:Number}
    m, n = size(H)

    @assert m == n

    # Compute the nonzero entries of p = (H - μI)e₁.
    @inbounds H₁₁ = H.dv[from]
    @inbounds H₂₁ = H.ev[from]

    p₁ = H₁₁ - μ
    p₂ = H₂₁

    # Map that column to a multiple of e₁ via two Given's rotations
    G₁, nrm = get_rotation(p₁, p₂, from)

    # Apply the Given's rotations
    @inbounds H₂₂ = H.dv[from+1]

    # The bulge-inducing element
    H₂₃ = from + 2 ≤ n ? @inbounds(H.ev[from + 1]) : zero(Tv)

    ## lmul!(G₁, H, from, min(from + 2, n))
    ## rmul!(H, G₁, from, min(from + 2, m))

    # First col
    H₁₁′ =  G₁.c * H₁₁ + G₁.s * H₂₁
    H₂₁′ = -G₁.s * H₁₁ + G₁.c * H₂₁

    # Second col
    H₁₂′ =  G₁.c * H₂₁ + G₁.s * H₂₂
    H₂₂′ = -G₁.s * H₂₁ + G₁.c * H₂₂

    # Third col
    H₁₃′ = G₁.s * H₂₃
    H₂₃′ = G₁.c * H₂₃

    # First row
    H₁₁′′ = H₁₁′ *  G₁.c + H₁₂′ * G₁.s # H₁₁ and H₁₂ touched by left mul
    H₁₂′′ = H₁₁′ * -G₁.s + H₁₂′ * G₁.c # H₁₁ and H₁₂ touched by left mul
    
    # Second row
    # H₂₁′′ = H₂₁′ *  G₁.c + H₂₂′ * G₁.s # H₂₁ and H₂₂ touched by left mul <-- no need to compute because we have H₁₂′′
    H₂₂′′ = H₂₁′ * -G₁.s + H₂₂′ * G₁.c # H₂₁ and H₂₂ touched by left mul
    
    # Third row
    # H₃₁′ = H₂₃ * G₁.s # H₃₁ = 0 and H₃₂ = H₂₃ <-- no need to compute because we have H₁₃′
    # H₃₂′ = H₂₃ * G₁.c # H₃₁ = 0 and H₃₂ = H₂₃ <-- no need to compute because we have H₂₃′

    # Store.
    H.dv[from + 0] = H₁₁′′
    H.dv[from + 1] = H₂₂′′
    H.ev[from + 0] = H₁₂′′

    # Conditonally store depending on whether the off-diagonal is still inbounds
    if from + 2 ≤ n
        H.ev[from + 1] = H₂₃′
    end
    
    # This is the value in the bulge bit that is on the second off-diagonal.
    bulge_value = H₁₃′

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

    @inbounds for i = from + 1 : to - 1
        
        p₁ = H.ev[i - 1]
        p₂ = bulge_value

        G, nrm = get_rotation(p₁, p₂, i)

        # Unbulging, we zero'd out the second off-diagonal
        # and we only store the off-diaognal
        H.ev[i - 1] = nrm
        
        # Rotate remaining columns
        # lmul!(G, H, i, min(i + 2, n))

        # B for bulge. Load the relevant values
        B₁₁ = H.dv[i + 0]
        B₂₂ = H.dv[i + 1]
        B₂₁ = H.ev[i + 0]

        # The bulge-inducing element
        B₂₃ = i + 2 ≤ n ? @inbounds(H.ev[i + 1]) : zero(Tv)

        # B₁₁ and B₂₁ is the first non-trivial col to apply the rotation on
        B₁₁′ =  G.c * B₁₁ + G.s * B₂₁
        B₂₁′ = -G.s * B₁₁ + G.c * B₂₁

        # Second col
        B₁₂′ =  G.c * B₂₁ + G.s * B₂₂ # B₂₁ = B₁₂
        B₂₂′ = -G.s * B₂₁ + G.c * B₂₂ # B₂₁ = B₁₂

        # Third col
        B₁₃′ = G.s * B₂₃
        B₂₃′ = G.c * B₂₃

        # First row
        B₁₁′′ = B₁₁′ *  G.c + B₁₂′ * G.s # B₁₁ and B₁₂ touched by left mul
        B₁₂′′ = B₁₁′ * -G.s + B₁₂′ * G.c # B₁₁ and B₁₂ touched by left mul
        
        # Second row
        # B₂₁′′ = B₂₁′ *  G.c + B₂₂′ * G.s # B₂₁ and B₂₂ touched by left mul <-- no need to compute because we have B₁₂′′
        B₂₂′′ = B₂₁′ * -G.s + B₂₂′ * G.c # B₂₁ and B₂₂ touched by left mul
        
        # Third row
        # B₃₁′ = B₂₃ * G.s # B₃₁ = 0 and B₃₂ = B₂₃ <-- no need to compute because we have B₁₃′
        # B₃₂′ = B₂₃ * G.c # B₃₁ = 0 and B₃₂ = B₂₃ <-- no need to compute because we have B₂₃′

        # Store.
        H.dv[i + 0] = B₁₁′′
        H.dv[i + 1] = B₂₂′′
        H.ev[i + 0] = B₁₂′′

        # Conditonally store depending on whether the off-diagonal is still inbounds
        if i + 2 ≤ n
            H.ev[i + 1] = B₂₃′
        end
        
        # This is the value in the bulge bit that is on the second off-diagonal.
        bulge_value = B₁₃′

        rmul!(Q, G)
    end

    H
end

function local_schurfact!(H::SymTridiagonal{T}, start::Int, to::Int, Q = NotWanted(), tol = eps(T), maxiter = 100*size(H, 1)) where {T<:Real}
    # iteration count
    iter = 0

    @inbounds while true

        if iter > maxiter
            return false, iter
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

            H₁₁, H₁₂ = H.dv[to-1], H.ev[to-1]
            H₂₁, H₂₂ = H.ev[to-1], H.dv[to]

            # Scaling to avoid losing precision in the case where we have nearly
            # repeated eigenvalues.
            scale = abs(H₁₁) + abs(H₁₂) + abs(H₂₁) + abs(H₂₂)
            H₁₁ /= scale; H₁₂ /= scale; H₂₁ /= scale; H₂₂ /= scale

            # Trace and discriminant of small eigenvalue problem.
            t = (H₁₁ + H₂₂) / 2
            d = (H₁₁ - t) * (H₂₂ - t) - H₁₂ * H₂₁
            sqrt_discr = sqrt(abs(d))

            # Very important to have a strict comparison here!
            @assert d < zero(T)

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
            single_shift_schur_tri!(H, from, to, λ, Q)
            iter += 1
        end

        # Converged!
        to ≤ start && break
    end

    return true, iter
end

###
### Real arithmetic
###
function local_schurfact!(H::AbstractMatrix{T}, start::Int, to::Int, 
                          Q = NotWanted(), tol = eps(T), 
                          maxiter = 100*size(H, 1)) where {T<:Real}
    # iteration count
    iter = 0

    @inbounds while true

        if iter > maxiter
            return false, iter
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
            H[from,from-1] = zero(T)
            H[from-1,from] = zero(T)
            to -= 1
        else
            # Now we are sure we can work with a 2×2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁, H₁₂ = H[to-1,to-1], H[to-1,to]
            H₂₁, H₂₂ = H[to  ,to-1], H[to  ,to]

            # Scaling to avoid losing precision in the case where we have nearly
            # repeated eigenvalues.
            scale = abs(H₁₁) + abs(H₁₂) + abs(H₂₁) + abs(H₂₂)
            H₁₁ /= scale; H₁₂ /= scale; H₂₁ /= scale; H₂₂ /= scale

            # Trace and discriminant of small eigenvalue problem.
            t = (H₁₁ + H₂₂) / 2
            d = (H₁₁ - t) * (H₂₂ - t) - H₁₂ * H₂₁
            sqrt_discr = sqrt(abs(d))

            # Very important to have a strict comparison here!
            @assert d < zero(T)

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
            single_shift_schur_tri!(H, from, to, λ, Q)
            iter += 1
        end

        # Converged!
        to ≤ start && break
    end

    return true, iter
end

local_schurfact!(H::AbstractMatrix{T}, Q = NotWanted(), tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T} =
    local_schurfact!(H, 1, size(H, 2), Q, tol, maxiter)