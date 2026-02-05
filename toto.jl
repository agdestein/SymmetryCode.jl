using WGLMakie
using LinearAlgebra
using FFTW

n = 50
x = range(0, 2π, 2n + 1)[2:end]
y = @. 10 * sin(6x)
dy = @. 60 * cos(6x)
yhat = rfft(y)
k = 0:n
dyhat = @. im * k * yhat
dyhat_inv = irfft(dyhat, 2n)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
    lines!(ax, x, dy; label = "True")
    lines!(ax, x, dyhat_inv; label = "FFT")
    axislegend(ax)
    fig
end
