using SymmetryCode
using Test

@testset "SymmetryCode" begin
    include("test_grid.jl")
    include("test_fft.jl")
    include("test_solver_kernels.jl")
    include("test_tensor_helpers.jl")
    include("test_symmetry.jl")
    include("test_closures.jl")
end
