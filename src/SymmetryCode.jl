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

include("solver.jl")
include("filtering.jl")
include("symmetry.jl")
include("nets.jl")
include("closures.jl")
include("data.jl")
include("les.jl")
include("training.jl")
include("analysis.jl")
include("plots.jl")
include("setups.jl")
include("experiment.jl")
include("verify.jl")

end
