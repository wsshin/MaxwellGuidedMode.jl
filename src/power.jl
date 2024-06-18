export poynting, powerₗ

# The function Re{⋅} is linear for real coefficients.  In other words, for complex numbers
# z = x + iy and w = u + iv,
#
# Re{p z + q w} = Re{p(x+iy) + q(u+iv)} = p x + q u = p Re{z} + q Re{w}.
#
# Therefore, when averaging the E- and H-fields for the Poynting vector calculation, the
# weight factors ∆l shouldn't contain the PML scale factors (i.e., we should not use s∆l).
# As long as we use real ∆l, we can average the Poynting vector before the last step.  In
# other words, from
#
# S = E×H＊ + E＊×H = 2 Re[E×H＊],
#
# we have
#
# Sz = 2 Re[Ex Hy＊ - Ey Hx＊] = 2 Re[Ex Hy＊] - 2 Re[Ey Hx＊].
#
# Among the two factors inside each Re[], I remember the averaging operation is applied to only and some of the averaged Cartesian components of the fields in this formula can be
# averaged
#
# Also note that the time-averaged Poynting vector is meaningful only for real frequency ω.
# This is usually not a problem because
# - in source-driven problems we use real frequency;
# - in waveguide mode problems we use ω-to-β formulation, where we use real ω and calculate
# complex β.
# However, note that in SALT we deal with complex ω below the lasing threshold.  Thankfully
# we don't need to calculate the Poynting vector for such nonlasing modes.
#
# I don't care about eliminating array allocations here, because the Poynting vector
# calculation is much cheaper than solution calculation.  If I really need to eliminate
# allocations, consider using for loops; otherwise such elimination will be difficult
# because of field averaging.
#
# For simplicity, always calculate the Poynting vectors as quantities entering the E-field
# voxels.
#
# On the Yee grid, the different Cartesian components of different vector fields are defined
# at unique locations.  Positioning these Cartesian components of vector fields at their
# correct locations (with the correct half-grid-point offsets) can be confusing, because
# there are quite a few different Yee grid types.  The 3D Yee grid for the 3D Maxwell
# equations is the most standard configuration, so the positions of the Cartesian components
# of the E- and H-fields in there are relatively well-known.  But in the grid configurations
# for other types of equations, such as the 2D TE and TM Maxwell equations and even the
# waveguide mode equation, the correct positions of the Cartesian components of the E- and H-
# fields can be confusing.  The correct positions of the Cartesian components of the
# Poyinting vector are even more confusing, because their calculation involves cross
# products and averaging.  So, the users must make sure to know where their vector fields
# are defined.  In the future, I may need to provide some wrapper functions that help the
# users on this regard.
function poynting(
    E::NTuple{Kₑ,AbstractArray{<:Number,K}},
    H::NTuple{Kₘ,AbstractArray{<:Number,K}},
    mdl::Model{K,Kₑ,Kₘ}
) where {K,Kₑ,Kₘ}
    S = poynting_complex(E, H, mdl)
    add_cc!.(S)

    return S
end

# Used in poynting().
function interp_field!(
    Gᵤ::AbstractArray{ComplexF64,K},  # where averaged fields are stored
    Fᵤ::AbstractArray{ComplexF64,K},  # field to average
    ft::AbstractFieldType,  # type of field to average
    nw::Int,  # direction of averaging
    ∆l::NTuple{2,NTuple{K,Vector{Float64}}},
    ∆l⁻¹::NTuple{2,NTuple{K,Vector{Float64}}},
    boundft::SVector{K,AbstractFieldType},
    isbloch::SVector{K,Bool};
    kwargs...  # keyword arguments for apply_m̂!()
) where {K}
    ft′ = conj(ft)
    isfwd = boundft[nw]==ft
    apply_m̂!(Gᵤ, Fᵤ, nw, isfwd, ∆l[ft][nw], ∆l⁻¹[ft′][nw], isbloch[nw]; kwargs...)

    return nothing
end

# Calculate the power along the longitudinal direction.
powerₗ(E, H, mdl) = add_cc(powerₗ_complex(E, H, mdl))

# This is needed for calculating the coefficient of the mth mode in a wave with E:
# Cₘ = ∫∫da ẑ⋅(E×conj(hₘ)) / ∫∫da ẑ⋅(eₘ×conj(hₘ)).
function powerₗ_complex(
    E::NTuple{Kₑ,AbstractArray{<:Number,K}},
    H::NTuple{Kₘ,AbstractArray{<:Number,K}},
    mdl::Model{K,Kₑ,Kₘ}
) where {K,Kₑ,Kₘ}
    S_complex = poynting_complex(E, H, mdl)
    Sₗ_complex = S_complex[end]

    gtₑ = gt4ft.(Hfield(), mdl.boundft)
    lvxl = getindex_col(mdl.grid.ghosted.lg, gtₑ)  # locations of x-normal edges of E-field pixels surrounding Sz

    vc = VoxelwiseConstant(Sₗ_complex, lvxl)
    Pₗ_complex = integral(vc)

    return Pₗ_complex
end

# Normalize the fields such that ∫∫da ẑ⋅E×conj(H) = 1/2.
# See RN - POWER > [Jun/02/2024] Mode coefficients calculation
function normalize!(
    E::NTuple{Kₑ,AbstractArray{<:Number,K}},
    H::NTuple{Kₘ,AbstractArray{<:Number,K}},
    mdl::Model{K,Kₑ,Kₘ}
) where {K,Kₑ,Kₘ}
    Pₗ = powerₗ(E, H, mdl)
    denom = √Pₗ

    for k = 1:Kₑ
        E[k] ./= denom
    end

    for k = 1:Kₘ
        H[k] ./= denom
    end

    # Now, powerₗ_complex(E, H, mdl) returns 1/2 if the waveguide is material-lossless.
    # See prototype/model/proto_full.m for example.

    return nothing
end
