module SymmetryCode

using Adapt
using CairoMakie
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
using Seneca
using StaticArrays
using Statistics
using Zygote

"Free up GPU memory."
function clean()
    GC.gc()
    CUDA.reclaim()
end
export clean

include("octahedral.jl")
include("spectral.jl")
include("setups.jl")

end
