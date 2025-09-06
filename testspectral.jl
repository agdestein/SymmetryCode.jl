if false
    include("src/SymmetryCode.jl")
    using .SymmetryCode
    using .SymmetryCode.Spectral
end

using LinearAlgebra
using Random
using Seneca
using SymmetryCode
using SymmetryCode.Spectral
using WGLMakie

dns_aid()

data = create_data()

data[2].xx .|> abs |> extrema
