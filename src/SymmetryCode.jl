module SymmetryCode

using Adapt
using CairoMakie
using ComponentArrays: ComponentArray
using CUDA
using FFTW
using JLD2
using KernelAbstractions
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using Lux
using Makie
using MLUtils
using Optimisers
using Random
using StaticArrays
using Statistics
using Zygote

"Free up GPU memory."
function clean()
    GC.gc()
    CUDA.functional() && CUDA.reclaim()
end
export clean

include("Seneca.jl")
export Seneca

using .Seneca

include("octahedral.jl")
include("spectral.jl")
include("training.jl")
include("setups.jl")

if false
    include("../testspectral.jl")
end

end
