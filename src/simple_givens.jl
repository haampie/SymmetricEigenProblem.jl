using LinearAlgebra: givensAlgorithm
using Base: @propagate_inbounds

import LinearAlgebra: lmul!, rmul!

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