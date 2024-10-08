using Plots, PyPlot
using Pkg
pyplot()

c = -0.4 + 0.61im
# c = 0.285 + 0.01im
# c = −0.835 − 0.2321im
# c = -0.8 + 0.156im
# c = −0.70176 + 0.3842im

R = 2
N = 1000
L = 1500
K = 1000

x = range(-1.5, 1.5; length=L)
y = range(-1.0, 1.0; length=K)
A = zeros(K, L)

function juliaset(z, c, R, N)
    n = 0
    while n <= N && abs(z) <= R^2 - R
        n += 1
        z = z^2 + c
    end
    return n > N ? 0 : n/N
end

function exercise_1()
    for k in 1:K, l in 1:L
        z = x[l] + y[k]*im
        for n in 0:N
            if abs(z) > R^2 - R
                A[k, l] = n/N
                break
            end
            z = z^2 + c
        end
    end
end

function draw_ex_1()
    heatmap(A;
        c=:viridis,
        clims=(0, 0.15),
        cbar=:none,
        axis=:none,
        ticks=:none
    )
end


cs = 0.7885 .* exp.(range(π/2, 3π/2; length = 500) .* im)
anim = @animate for c in cs
    A = juliaset.(x' .+ y .* im, c, R, N)
    heatmap(A;
        c = :viridis,
        clims = (0, 0.15),
        cbar = :none,
        axis = :none,
        ticks = :none,
        size = (800, 600),
    )
end
gif(anim, "juliaset.gif", fps = 20) # save animation as a gif
