using MaxwellGuidedMode
using SimpleConstants
using Arpack
using CairoMakie

## Set the oscillation parameters.
L₀ = 1e-9  # nm
unit = MaxwellUnit(L₀)
λ = 1030.0  # 30L₀
# λ = 600.0
ω = 2πc₀ / (λ*L₀) / unit.ω

## Create a grid.
r = 2000.0  # "radius" in ∞-norm
∆r = 5.0
xprim = collect(-r:∆r:r)

grid = Grid(tuple(xprim), [false])  # isbloch = false; important for narrower band
N = grid.N

## Create a model
mdl = MaxwellGuidedMode.ModelTE(grid, APtype=Array{Float64,3})

Npml = ([0], [0])
set_ωpml!(mdl, ω)
set_Npml!(mdl, Npml)

## Add objects.
εair = 1.0

ncore = 1.9; εcore = ncore^2

w = 800

add_obj!(mdl, "Air", Cuboid(grid.bounds), ε=εair)
add_obj!(mdl, "Core", Cuboid([0], [w]), ε=εcore)

## Create the eigenequation and solve it.
Ps = create_paramops(mdl)
∇̽s = create_curls(mdl)
πcmps = create_πcmps(mdl)

A = create_A(EE, ω, Ps, ∇̽s, πcmps)
@show eltype(A)<:Real  # isreal(A) can be true even if eltype(A) <: Complex if imaginary part is zero; isreal(A) is also very slow because it tests all entries
βguess = 2π * ncore / λ

# I know A is not symmetric.  But is it at least positive-definite?
# Below, if I try sigma=-1.0, I do get β² ≈ -0.08.  norm(A*e - β².*e) ≈ τₐ₀, so this
# negative eigenvalue is legit.  So, A is neither symmetric nor positive-definite...
@time β², e = eigs(A, nev=6, sigma=βguess^2, maxiter=1000)
# @time γ², f = eigs(A; nev=10, which=:SM)
β = .√(β²)

## Other components
k = 1
eₜ = e[:,k]
β = .√β²[k]

Etot, Htot = complete_fields(β, eₜ, EE, ω, Ps, ∇̽s, πcmps, mdl)
Ey = only(Etot)
Hx, Hz = Htot

Emax = maximum(abs, Ey)

Hnorm = .√(+).(abs2.(Hx), abs2.(Hz))
Hmax = maximum(Hnorm)
Hx_max = maximum(abs, Hx)
Hz_max = maximum(abs, Hz)
println("Hx_max/Hmax = $(Hx_max/Hmax), Hz_max/Hmax = $(Hz_max/Hmax)")

## Plot the solution.
nX = 1

gtₑ = ft2gt.(mdl.boundft[nX], EE)
gtₘ = ft2gt.(mdl.boundft[nX], HH)

xₑ = grid.l[Int(gtₑ)][nX]
xₘ = grid.l[Int(gtₘ)][nX]

fig = Figure()

Axis(fig[1,1], title="Ey")
lines!(xₑ, real.(Ey) ./ Emax)

Axis(fig[2,1], title="Hx")
lines!(xₑ, real.(Hx) ./ Hmax)  # xₑ, becuase Hx is co-located with Ey

Axis(fig[3,1], title="Hz")
lines!(xₘ, real.(Hz) ./ Hmax)

display(fig)

## Poynting vector
Stot = poynting(Etot, Htot, mdl)
Sx, Sz = Stot
Snorm = .√(+).(abs2.(Sx), abs2.(Sz))
Smax = maximum(Snorm)

fig = Figure()

Axis(fig[1,1], title="Sx")
lines!(xₑ, abs.(Sx) ./ Smax)

Axis(fig[2,1], title="Sz")
lines!(xₘ, abs.(Sz) ./ Smax)

display(fig)
