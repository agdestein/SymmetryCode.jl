module SymmetryCode

using AbstractFFTs
using Adapt
using CUDA
using CairoMakie
using ComponentArrays: ComponentArray
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

default_backend() = CUDA.functional() ? CUDABackend() : CPU()

"Free up GPU memory."
function clean()
    GC.gc()
    CUDA.functional() && CUDA.reclaim()
    return
end

include("Seneca.jl")
include("filtering.jl")
include("symmetry.jl")
include("octahedral.jl")
include("spectral.jl")
include("training.jl")
include("setups.jl")
include("verify.jl")

end
