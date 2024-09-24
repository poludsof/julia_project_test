using Plots

n = 1000
t = range(0, 2π; length = n)
linewidth1 = range(1, 50; length = 500)
linewidth2 = range(50, 1; length = 500)
linewidth = vcat(linewidth1, linewidth2)

color = palette(:viridis, 2)
xlims = (0, 7)
ylims = (-1.2, 1.2)
label = ""

fx(t) = cos(3t)
fy(t) = sin(2t)

color = vcat(
    collect(palette(:viridis, n ÷ 2)),
    collect(palette(:viridis, n - n ÷ 2; rev = true))
)

# Plots.plot(t, cos.(3*t); linewidth, color, label, xlims, ylims)
# Plots.plot(t, sin.(2*t)); linewidth, color, label, xlims, ylims)
Plots.plot(fx.(t), fy.(t);
    linewidth,
    color,
    lims = (-1.2, 1.2),
    legend = false,
    axis = nothing,
    border = :none,
)


a = 4.23
b = 2.35
t = range(-15, 20; length = 500)
fx(t) = (a + b) * cos(t) - b*cos(((a/b) + 1) * t)
fy(t) = (a + b) * sin(t) - b*sin(((a/b) + 1) * t)
Plots.plot(fx, fy, -15, 20, 500; linewidth = 2)