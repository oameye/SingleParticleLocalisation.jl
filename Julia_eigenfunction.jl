using PyPlot
using LinearAlgebra
using SparseArrays
using Arpack

function meshgrid(x::LinRange{Float64, Int64}, y::LinRange{Float64, Int64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]
    X, Y
end

function get_potential(x::Float64, y::Float64, V₀::Float64)::Float64
    return V₀*(cos(pi*x)^2 + cos(pi*√2/2*(x+y))^2 + cos(pi*y)^2 + cos(-pi/(√2)*(x-y))^2)
end

function eigenfunction(V::Matrix{Float64})
    N = size(V)[1]
    # creates the discretised 2nd derivative
    D = sparse(Tridiagonal(ones(N-1), -2*ones(N), ones(N-1)))
    # N**2 x N**2 matrix
    T = -1/2 * (kron(D, sparse(I,N,N)) + kron(sparse(I,N,N), D))
    U = spdiagm(reshape(V, N^2))
    H = T + U;

    _, eigenvector = eigs(H, nev=1, which=:SM);

    return reshape(eigenvector', N, N)
end
const V₀ = 1.0
const L = 20.0
const N = 150
X, Y = meshgrid(LinRange(-L, L, N), LinRange(-L, L, N));
V = get_potential.(X, Y, V₀)
fig = figure(figsize=(7,7))
contourf(X, Y, V)
ef = eigenfunction(V)
pygui(true)
figure(figsize=(9,9))
contourf(X, Y, ef^2, 100)