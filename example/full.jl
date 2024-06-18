using MaxwellGuidedMode
using SimpleConstants
using Arpack
using CairoMakie
using VoxelwiseConstantMapping

## Set the oscillation parameters.
L₀ = 1e-9  # nm
unit = MaxwellUnit(L₀)
λ = 1030.0  # 30L₀
# λ = 600.0
ω = 2πc₀ / (λ*L₀) / unit.ω

## Create a grid.
r = 2000.0  # "radius" in ∞-norm
∆r = 10.0
xprim = collect(-r:∆r:r)
yprim = collect(-r:∆r:r)  # use -2r:0.1:2r for better unidirectional emission

grid = Grid((xprim,yprim), [false,false])  # isbloch = false; important for narrower band
N = grid.N
## Create a model
mdl = MaxwellGuidedMode.ModelFull(grid, APtype=Array{Float64,4})

Npml = ([0,0], [0,0])
set_ωpml!(mdl, ω)
set_Npml!(mdl, Npml)

## Add objects.
εair = 1.0

ncore = 1.9; εcore = ncore^2
nsubs = 1.45; εsubs = nsubs^2

w = 800
h = 500

add_obj!(mdl, "Air", Cuboid(grid.bounds), ε=εair)
add_obj!(mdl, "Substrate", Cuboid(([-r,-r], [r,-h/2])), ε=εsubs)
add_obj!(mdl, "Core", Cuboid([0,0], [w,h]), ε=εcore)

## Create the eigenequation and solve it.
Ps = create_paramops(mdl)
∇̽s = create_curls(mdl)
πcmps = create_πcmps(mdl)

# A, A₁₁, A₁₂, A₂₁, A₂₂ = create_A(Hfield(), ω, Ps, ∇̽s, πcmps)
A = create_A(Efield(), ω, Ps, ∇̽s, πcmps)
@show eltype(A)<:Real  # isreal(A) can be true even if eltype(A) <: Complex if imaginary part is zero; isreal(A) is also very slow because it tests all entries
βguess = 2π * ncore / λ
# Fguess = zeros(2, N...)
# Fguess[1,:,:] .= 1
# fguess = reshape(Fguess, :, 1)[:,1]

# I know A is not symmetric.  But is it at least positive-definite?
# Below, if I try sigma=-1.0, I do get β² ≈ -0.08.  norm(A*e - β².*e) ≈ τₐ₀, so this
# negative eigenvalue is legit.  So, A is neither symmetric nor positive-definite...
@time β², e = eigs(A, nev=1, sigma=βguess^2)
# @time β², f = eigs(A; nev=10, which=:SM)
# @time β, f = calc_mode(A)

## Other components
k = 1
eₜ = e[:,k]
β = .√β²[k]

Etot, Htot = complete_fields(β, eₜ, Efield(), ω, Ps, ∇̽s, πcmps, mdl)
normalize!(Etot, Htot, mdl)

Pₗ_complex = MaxwellGuidedMode.powerₗ_complex(Etot, Htot, mdl)
Pₗ = powerₗ(Etot, Htot, mdl)

@show Pₗ_complex, Pₗ

Ex, Ey, Ez = Etot
Hx, Hy, Hz = Htot

# Below, different Cartesian components are defined at diffeernt locations, so the Enorm
# calculation is not accurate.
Enorm = .√(+).(abs2.(Ex), abs2.(Ey), abs2.(Ez))
Emax = maximum(Enorm)
Ex_max = maximum(abs, Ex)
Ey_max = maximum(abs, Ey)
Ez_max = maximum(abs, Ez)

Hnorm = .√(+).(abs2.(Hx), abs2.(Hy), abs2.(Hz))
Hmax = maximum(Hnorm)
Hx_max = maximum(abs, Hx)
Hy_max = maximum(abs, Hy)
Hz_max = maximum(abs, Hz)

# Ez_max is pretty large.
println("Ex_max/Emax = $(Ex_max/Emax), Ey_max/Emax = $(Ey_max/Emax), Ez_max/Emax = $(Ez_max/Emax)")
println("Hx_max/Hmax = $(Hx_max/Hmax), Hy_max/Hmax = $(Hy_max/Hmax), Hz_max/Hmax = $(Hz_max/Hmax)")

## Visualize.
# Note that the positive-end boundary will not be shown, because the data points there are
# ghost points.  The X and Y arrays passed to pcolormesh describe pixel boundaries, and for
# Ez they are dual grid plane locations.  Even if we take the ghosted l for xs and ys above,
# note that they still don't have the right boundary of the positive-x-end pixel and top
# boundary of the positive-y-end pixel.  That is why the following figures do not show the
# ticks for the positive-end boundaries in x and y.
nX, nY = 1, 2

gₑ = gt4ft.(mdl.boundft, Efield())
gₘ = gt4ft.(mdl.boundft, Hfield())

gₑx = gₑ[nX]
gₑy = gₑ[nY]

gₘx = gₘ[nX]
gₘy = gₘ[nY]

xₑ = grid.l[gₑx][nX]
yₑ = grid.l[gₑy][nY]

xₘ = grid.l[gₘx][nX]
yₘ = grid.l[gₘy][nY]

fig = Figure()

ax = Axis(fig[1,1], title="Ex")
ax.aspect = DataAspect()
heatmap!(xₘ, yₑ, abs.(Ex) ./ Emax, colormap=:hot, colorrange=(0,1))

ax = Axis(fig[1,2], title="Ey")
ax.aspect = DataAspect()
heatmap!(xₑ, yₘ, abs.(Ey) ./ Emax, colormap=:hot, colorrange=(0,1))

ax = Axis(fig[1,3], title="Ez")
ax.aspect = DataAspect()
heatmap!(xₑ, yₑ, abs.(Ez) ./ Emax, colormap=:hot, colorrange=(0,1))

ax = Axis(fig[2,1], title="Hx")
ax.aspect = DataAspect()
heatmap!(xₑ, yₘ, abs.(Hx) ./ Hmax, colormap=:hot, colorrange=(0,1))

ax = Axis(fig[2,2], title="Hy")
ax.aspect = DataAspect()
heatmap!(xₘ, yₑ, abs.(Hy) ./ Hmax, colormap=:hot, colorrange=(0,1))

ax = Axis(fig[2,3], title="Hz")
ax.aspect = DataAspect()
heatmap!(xₘ, yₘ, abs.(Hz) ./ Hmax, colormap=:hot, colorrange=(0,1))

# Colorbar(fig[1,4], hm3)
display(fig)

# # It seems that the modal fields are all in phase, so the real part of the fields seems to
# # be the same as the modulus of the fields.
# fig2 = Figure()
# ax2, hm2 = heatmap(fig2[1,1][1,1], xs, ys, real.(Ex) ./ Emax, colormap=:bwr, colorrange=(-1,1))
# ax2.aspect = DataAspect()
# Colorbar(fig2[1,1][1,2], hm2)
# display(fig2)

## Poynting vector
Stot = poynting(Etot, Htot, mdl)
Sx, Sy, Sz = Stot
Snorm = .√(+).(abs2.(Sx), abs2.(Sy), abs2.(Sz))
Smax = maximum(Snorm)

fig = Figure()

ax = Axis(fig[1,1], title="Sx")
ax.aspect = DataAspect()
heatmap!(xₑ, yₘ, abs.(Sx) ./ Smax, colormap=:bwr, colorrange=(-1,1))

ax = Axis(fig[1,2], title="Sy")
ax.aspect = DataAspect()
heatmap!(xₘ, yₑ, abs.(Sy) ./ Smax, colormap=:bwr, colorrange=(-1,1))

ax = Axis(fig[1,3], title="Sz")
ax.aspect = DataAspect()
heatmap!(xₘ, yₘ, abs.(Sz) ./ Smax, colormap=:bwr, colorrange=(-1,1))

display(fig)

xgₑ = grid.ghosted.lg[gₑx][nX]
ygₑ = grid.ghosted.lg[gₑy][nY]

Sz_pw = VoxelwiseConstant(Sz, (xgₑ,ygₑ))
