using LinearAlgebra
using LinearAlgebra: givensAlgorithm 
using Base: OneTo

abstract type SimpleRotation end

"""
When moving a bulge of size three two steps forward, one gets in total four rotations
hitting only four rows or columns, which means six flops per memop and everything fitting
into registers. This seems optimal for Float64 with AVX.
"""
struct Rotation3x2{T} <: SimpleRotation
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

struct Rotation3{T} <: SimpleRotation
    c1::T
    s1::T
    c2::T
    s2::T
    i::Int
end

@inline function rmul_kernel(a0, a1, a2, a3, G::Rotation3x2)
    # Move the bulge from 0 → 1
    a1′    = muladd( a1, G.c1, a2 * G.s1)
    a2′    = muladd(-a1, G.s1, a2 * G.c1)
    a0′′   = muladd( a0, G.c2, a1′ * G.s2)
    a1′′   = muladd(-a0, G.s2, a1′ * G.c2)
    
    # Move the bulge from 1 → 2
    a2′′′  = muladd( a2′, G.c3, a3 * G.s3)
    a3′′′  = muladd(-a2′, G.s3, a3 * G.c3 )
    a1′′′′ = muladd( a1′′, G.c4, a2′′′ * G.s4)
    a2′′′′ = muladd(-a1′′, G.s4, a2′′′ * G.c4)

    return a0′′, a1′′′′, a2′′′′, a3′′′
end

@inline function rmul_kernel(a0, a1, a2, G::Rotation3)
    # Move the bulge from 0 → 1
    a1′    = muladd( a1, G.c1, a2 * G.s1)
    a2′    = muladd(-a1, G.s1, a2 * G.c1)
    a0′′   = muladd( a0, G.c2, a1′ * G.s2)
    a1′′   = muladd(-a0, G.s2, a1′ * G.c2)

    return a0′′, a1′′, a2′
end

@inline rmul!(Q, G::SimpleRotation) = rmul!(Q, G, axes(Q, 1))

function rmul!(Q, G::Rotation3x2, range)
    @inbounds for j = range
        a0 = Q[j, G.i + 0]
        a1 = Q[j, G.i + 1]
        a2 = Q[j, G.i + 2]
        a3 = Q[j, G.i + 3]

        a0′, a1′, a2′, a3′ = rmul_kernel(a0, a1, a2, a3, G)

        Q[j, G.i + 0] = a0′
        Q[j, G.i + 1] = a1′
        Q[j, G.i + 2] = a2′
        Q[j, G.i + 3] = a3′
    end
end

function rmul!(Q, G::Rotation3, range)
    @inbounds for j = range
        a0 = Q[j, G.i + 0]
        a1 = Q[j, G.i + 1]
        a2 = Q[j, G.i + 2]

        a0′, a1′, a2′ = rmul_kernel(a0, a1, a2, G)

        Q[j, G.i + 0] = a0′
        Q[j, G.i + 1] = a1′
        Q[j, G.i + 2] = a2′
    end
end

generate_rotation() = givensAlgorithm(rand(), rand())[1:2]
generate_double_shift() = (generate_rotation()..., generate_rotation()...)
generate_rotations(bulges, steps) = [generate_double_shift() for b = OneTo(bulges), k = OneTo(steps)]

@inline colnumber(num_bulges::Int, bulge::Int, step::Int) = 3(num_bulges - bulge) + step

function trivial!(Q, rotations, offset::Int = 0)
    b = size(rotations, 1)

    @inbounds for y = axes(rotations, 1), x = axes(rotations, 2)
        c1, s1, c2, s2 = rotations[y, x]
        rmul!(Q, Rotation3(c1, s1, c2, s2, colnumber(b, y, x + offset)))
    end

    return Q
end

function wave!(Q, rotations, offset::Int = 0)
    b = size(rotations, 1)
    s = size(rotations, 2)

    # For simplicity -- although below is defo too complex
    @assert iseven(b) && iseven(s)

    # startup
    @inbounds for k = b:-2:1
        x, y = 1, k
        c1, s1, c2, s2 = rotations[y, x]
        rmul!(Q, Rotation3(c1, s1, c2, s2, colnumber(b, y, x + offset)))

        for l = k+1:b
            x, y = l - k, l
            c1, s1, c2, s2 = rotations[y, x + 0]
            c3, s3, c4, s4 = rotations[y, x + 1]
            rmul!(Q, Rotation3x2(c1, s1, c2, s2, c3, s3, c4, s4, colnumber(b, y, x + offset)))
        end
    end

    # pipeline
    @inbounds for k = 1:2:s-b, l = 1:b
        x, y = k-1+l, l
        c1, s1, c2, s2 = rotations[y, x + 0]
        c3, s3, c4, s4 = rotations[y, x + 1]
        c = colnumber(b, y, x + offset)
        rmul!(Q, Rotation3x2(c1, s1, c2, s2, c3, s3, c4, s4, colnumber(b, y, x + offset)))
    end

    # shutdown
    @inbounds for k = s-b:2:s-1
        for l = 1:s-k-1
            x, y = k+l, l
            c1, s1, c2, s2 = rotations[y, x + 0]
            c3, s3, c4, s4 = rotations[y, x + 1]
            rmul!(Q, Rotation3x2(c1, s1, c2, s2, c3, s3, c4, s4, colnumber(b, y, x + offset)))
        end

        x, y = s, s - k

        c1, s1, c2, s2 = rotations[y, x]
        rmul!(Q, Rotation3(c1, s1, c2, s2, colnumber(b, y, x + offset)))
    end

    return Q
end

function blocked_wave!(Q, rotations, offset::Int = 0, blocksize::Int = 256)
    from = offset + 1
    to = offset + 3size(rotations, 1) + size(rotations, 2) - 1
    
    @assert size(Q, 1) % blocksize == 0

    panel = similar(Q, blocksize, length(from:to))

    @inbounds for i = 1 : blocksize : size(Q, 1)
        Qview = view(Q, i:i+blocksize - 1, from:to)
        copyto!(panel, Qview)
        wave!(panel, rotations, 0)
        copyto!(Qview, panel)
    end

    Q
end

using Test

"""
Test whether trivial order == wave order
"""
function test_stuff(bulges = 4, steps = 10)
    rotations = generate_rotations(bulges, steps)
    n = 3bulges + steps - 1
    Q1 = Matrix(1.0I, 4n, n)
    Q2 = copy(Q1)
    Q3 = copy(Q1)
    wave!(Q1, rotations)
    trivial!(Q2, rotations)
    blocked_wave!(Q3, rotations, 0, 4)

    @test Q1 ≈ Q2 ≈ Q3
end

## tools

"""
Makes a plot of a matrix like in the article
"""
function matrix_structure(A)
    nz = 0
    @inbounds for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            if iszero(A[i, j])
                print(". ")
            else
                print("x ")
                nz += 1
            end

            if j == size(A, 2) ÷ 2
                print("| ")
            end
        end
        println()
        if i == size(A, 2) ÷ 2
            println("- " ^ (size(A,1) ÷ 2), "+ ", "- " ^ (size(A,1) ÷ 2 + size(A,1)%2))
        end
    end
    println("Nonzero ratio: ", round.(nz / length(A), digits = 2))
    println("Size = ", size(A))
end

function example(bulges = 4, steps = 10)
    rotations = generate_rotations(bulges, steps)
    n = 3bulges + steps - 1
    return trivial!(Matrix(1.0I, n, n), rotations)
end

## benching

using BenchmarkTools

function flopcount(collength = 3_000, bulges = 6, steps = 3bulges)
    rotations = generate_rotations(bulges, steps)
    Q = rand(collength, 3bulges + steps)

    # one rotation = 6 flops
    # two rotations per entry
    # number of rotations = length(rotations)
    # applied to each row
    flop = collength * length(rotations) * 2 * 6

    time1 = @belapsed wave!($Q, $rotations, $(0))
    # time2 = @belapsed blocked_wave!($Q, $rotations, $(0), 128)

    return flop / time1 / 1e9#, flop / time2 / 1e9
end