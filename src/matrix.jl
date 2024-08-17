export create_paramops, create_curls, create_πcmps
export create_ⅈf′ₗ4fₜ, create_ⅈhₗ4eₜ, create_ⅈeₗ4hₜ  # ⅈ is entered by \bbii<tab>
export create_βf′ₜ4fₜ, create_βhₜ4eₜ, create_βeₜ4hₜ
export create_A

function create_stretched_∆ls(mdl::Model)
    @unpack ωpml, grid, Npml, boundft = mdl

    s∆l = create_stretched_∆l(ωpml, grid, Npml)
    s∆l⁻¹ = invert_∆l(s∆l)

    gtₑ = gt4ft.(Efield(), boundft)
    gtₘ = gt4ft.(Hfield(), boundft)

    s∆lₑ = getindex_col(s∆l, gtₑ)  # stretched ∆l's centered at E-field plane locations
    s∆lₘ = getindex_col(s∆l, gtₘ)  # stretched ∆l's centered at H-field plane locations
    s∆lₑ⁻¹ = getindex_col(s∆l⁻¹, gtₑ)
    s∆lₘ⁻¹ = getindex_col(s∆l⁻¹, gtₘ)

    return (s∆lₑ, s∆lₘ), (s∆lₑ⁻¹, s∆lₘ⁻¹)
end

function create_paramops(mdl::Model{K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}) where {K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}
    (s∆lₑ, s∆lₘ), (s∆lₑ⁻¹, s∆lₘ⁻¹) = create_stretched_∆ls(mdl)
    calc_matparams!(mdl)  # assignment and smoothing; implemented for each specialized alias of Model

    @unpack grid, boundft, εₜarr, μₜarr, εₗarr, μₗarr, cmpfirst = mdl
    isbloch = grid.isbloch

    e⁻ⁱᵏᴸ = create_e⁻ⁱᵏᴸ(mdl.kbloch, grid.L)

    isfwd_inₑ = boundft.!=Efield()
    isfwd_inₘ = boundft.!=Hfield()

    Pεₜ = Kₑₜ==1 ? create_paramop(εₜarr; cmpfirst) :
                  create_paramop(εₜarr, isfwd_inₑ, s∆lₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmpfirst)
    Pμₜ = Kₘₜ==1 ? create_paramop(μₜarr; cmpfirst) :
                  create_paramop(μₜarr, isfwd_inₘ, s∆lₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmpfirst)

    # The following lines work even if the model does not deal with the corresponding
    # longitudinal components.
    Pεₗ = create_paramop(εₗarr; cmpfirst)  # 0×0 matrix for Kₑₗ = 0 (no longitudinal E-field)
    Pμₗ = create_paramop(μₗarr; cmpfirst)  # 0×0 matrix for Kₘₗ = 0 (no longitudinal H-field)

    return (Pεₜ,Pμₜ), (Pεₗ,Pμₗ)
end

function create_curls(mdl::Model)
    _, (s∆lₑ⁻¹, s∆lₘ⁻¹) = create_stretched_∆ls(mdl)

    @unpack grid, boundft, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, cmpfirst = mdl
    isbloch = grid.isbloch

    e⁻ⁱᵏᴸ = create_e⁻ⁱᵏᴸ(mdl.kbloch, grid.L)

    isfwdₑ = boundft.==Efield()
    isfwdₘ = boundft.==Hfield()

    # The following lines work even if the model does not deal with the corresponding
    # longitudinal components.
    ∇̽ₑₜ = create_curl(isfwdₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_out=cmpₘₗ, cmp_in=cmpₑₜ, cmpfirst)
    ∇̽ₘₜ = create_curl(isfwdₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_out=cmpₑₗ, cmp_in=cmpₘₜ, cmpfirst)

    ∇̽ₑₗ = create_curl(isfwdₑ, s∆lₘ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_out=cmpₘₜ, cmp_in=cmpₑₗ, cmpfirst)
    ∇̽ₘₗ = create_curl(isfwdₘ, s∆lₑ⁻¹, isbloch, e⁻ⁱᵏᴸ; cmp_out=cmpₑₜ, cmp_in=cmpₘₗ, cmpfirst)

    return (∇̽ₑₜ,∇̽ₘₜ), (∇̽ₑₗ,∇̽ₘₗ)
end

function create_πcmps(mdl::Model{K}) where {K}
    @unpack grid, cmpₑₜ, cmpₘₜ, cmpfirst = mdl

    cmpperm = reverse(SVector{K}(1:K))
    scale₀ = SVector(1,-1)

    πcmpₑ = create_πcmp(grid.N, cmpperm; scale=scale₀[cmpₑₜ], cmpfirst)
    πcmpₘ = create_πcmp(grid.N, cmpperm; scale=scale₀[cmpₘₜ], cmpfirst)

    return πcmpₑ, πcmpₘ
end

create_ⅈhₗ4eₜ(
    ω,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}}
) =
    create_ⅈf′ₗ4fₜ(Efield(), ω, Ps, ∇̽s)

create_ⅈeₗ4hₜ(
    ω,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}}
) =
    create_ⅈf′ₗ4fₜ(Hfield(), ω, Ps, ∇̽s)

create_ⅈf′ₗ4fₜ(
    ft, ω,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}}
) =
    create_ⅈf′ₗ4fₜ(ft, ω, Ps[2], ∇̽s[1])

# Create the operator that generates im * F′ₗ from Fₜ, where F′ is the field complementary
# to F (e.g., F′ = H-field for F = E-field).
function create_ⅈf′ₗ4fₜ(
    ft::AbstractFieldType,  # type of input field Fₜ, not output field F′ₗ
    ω::Number,
    Pₗs::NTuple{2,AbstractMatrix{<:Number}},
    ∇̽ₜs::NTuple{2,AbstractMatrix{<:Number}}
)
    ft′ = conj(ft)

    α = ft==Efield() ? -1/ω : 1/ω

    P′ₗ = Pₗs[ft′]
    ∇̽ₜ = ∇̽ₜs[ft]

    L = α .* (P′ₗ \ ∇̽ₜ)

    return L
end


create_βhₜ4eₜ(ω, Ps, ∇̽s, πcmps) = create_βf′ₜ4fₜ(Efield(), ω, Ps, ∇̽s, πcmps)
create_βeₜ4hₜ(ω, Ps, ∇̽s, πcmps) = create_βf′ₜ4fₜ(Hfield(), ω, Ps, ∇̽s, πcmps)

# Create the operator that generates β * F′ₜ from Fₜ, where F′ is the field complementary to
# F (e.g., F′ = H-field for F = E-field).
create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps) = sum(create_βf′ₜ4fₜ_comps(ft, ω, Ps, ∇̽s, πcmps))

function create_βf′ₜ4fₜ_comps(
    ft::AbstractFieldType,  # type of input field Fₜ, not output field F′ₗ
    ω::Number,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    πcmps::NTuple{2,AbstractMatrix{<:Number}}
)
    ft′ = conj(ft)

    α = ft==Efield() ? ω : -ω

    Pₜs, Pₗs = Ps
    ∇̽ₜs, ∇̽ₗs = ∇̽s

    Pₜ = Pₜs[ft]
    ∇̽′ₗ = ∇̽ₗs[ft′]
    πcmp = πcmps[ft]

    βf′ₜ4fₜ₁ = (α .* πcmp) * Pₜ

    if length(Pₗs[ft′]) > 0
        ⅈf′ₗ4fₜ = create_ⅈf′ₗ4fₜ(ft, ω, Pₗs, ∇̽ₜs)
        βf′ₜ4fₜ₂ = (πcmp * ∇̽′ₗ) * ⅈf′ₗ4fₜ
    else
        βf′ₜ4fₜ₂ = spzeros(eltype(βf′ₜ4fₜ₁), size(βf′ₜ4fₜ₁)...)
    end

    return βf′ₜ4fₜ₁, βf′ₜ4fₜ₂
end

function create_A(
    ft::AbstractFieldType,  # type of input field Fₜ
    ω::Number,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    πcmps::NTuple{2,AbstractMatrix{<:Number}}
)
    ft′ = conj(ft)

    # Old implementation.  Though more straightforward, this implementation does not
    # gurantee the vector calculus identity βfₜ4f′ₜ₂ * βf′ₜ4fₜ₂ = 0 due to rounding errors.
    #
    # βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    # βfₜ4f′ₜ = create_βf′ₜ4fₜ(ft′, ω, Ps, ∇̽s, πcmps)
    # A = βfₜ4f′ₜ * βf′ₜ4fₜ

    βf′ₜ4fₜ₁, βf′ₜ4fₜ₂ = create_βf′ₜ4fₜ_comps(ft, ω, Ps, ∇̽s, πcmps)
    βfₜ4f′ₜ₁, βfₜ4f′ₜ₂ = create_βf′ₜ4fₜ_comps(ft′, ω, Ps, ∇̽s, πcmps)

    A = βfₜ4f′ₜ₁ * βf′ₜ4fₜ₁ + βfₜ4f′ₜ₁ * βf′ₜ4fₜ₂ + βfₜ4f′ₜ₂ * βf′ₜ4fₜ₁  # βfₜ4f′ₜ₂ * βf′ₜ4fₜ₂ = 0 always

    return A
end
