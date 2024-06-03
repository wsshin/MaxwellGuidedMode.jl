# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K²,K₊₂, P,AP} for the TE waveguide mode
# mode equations for 1D cross section.
#
# The type parameter AP specifies device-specific arrays types (e.g., CuArray) and is
# user-defined in the constructor.
#
# Note that the TE mode has (E₂, H₁, H₃) as nonzero components; this is unlike TE Maxwell's
# equations that have two E-field components and one H-field component.

const ModelTE{P,AP} = Model{1,1,2, 1,1, 0,1, 1,3, P,AP}

# Convenience constructor
function ModelTE(grid::Grid; APtype::Type=Array{ComplexF,3})
    ft_eq = EE

    cmpₑₜ = SVector(2)  # transverse E-field (2nd component)
    cmpₘₜ = SVector(1)  # transverse H-field (1st component)

    cmpₑₗ = SVector{0,Int}()  # longitudinal E-field (no component)
    cmpₘₗ = SVector(3)  # longitudinal H-field (3rd component)

    iseₜ˔shp = true
    ishₜ˔shp = false

    return ModelTE{eltype(APtype),APtype}(; ft_eq, grid, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelTE)
    # Take necessary fields of mdl.
    @unpack grid, boundft,
            εₜarr, µₜarr, µₗarr,
            shp4oind, εind4oind, μind4oind, ε4εind, μ4μind,
            cmpₑₜ, cmpₘₜ, cmpₘₗ,
            iseₜ˔shp, ishₜ˔shp = mdl

    @unpack isbloch, N, l, σ, ghosted = grid
    @unpack lg, τlg, ∆τ = ghosted

    gtₑ = ft2gt.(EE,boundft)
    gtₘ = ft2gt.(HH,boundft)

    εₜ4εind = sub_matprm4pind(ε4εind, cmpₑₜ)
    μₜ4μind = sub_matprm4pind(μ4μind, cmpₘₜ)
    μₗ4μind = sub_matprm4pind(μ4μind, cmpₘₗ)

    ishₗ˔shp = true

    # Create temporary storages.
    # Because E₂ and H₁ are at the same locations, smoothing them can use the same object
    # index.
    oind2d_pₜ = create_oind_array(N)  # for εₜ (= ε₂₂) and μₜ (= μ₁₁)
    oind2d_pₗ = create_oind_array(N)  # for μₗ (= μ₃₃)
    oind2d′_pₗ = create_oind_array(N)  # for @assert below

    # Assign the material parameters.
    # Note that μₜarr uses gtₑ rather than gtₘ, because it is rank-0 tensor and its locations
    # are the E-field locations; see the definition MaxwellBase/assign_param!().
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation; take the cross section through E₂, H₁, H₃.)
    assign_param!(εₜarr, tuple(oind2d_pₗ), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # εₜ tensors (rank-0, so scalars)
    assign_param!(µₜarr, tuple(oind2d′_pₗ), shp4oind, μind4oind, μₜ4μind, gtₑ, τlg, isbloch)  # μₜ tensors (rank-0, so scalars)
    assign_param!(µₗarr, tuple(oind2d_pₜ), shp4oind, μind4oind, μₗ4μind, gtₘ, τlg, isbloch)  # μₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  Specifically, oind2d_ε₂₂ and oind2d_μ₁₁
    # are evaluated at the locations surrounding the ε₂₂ and μ₁₁ locations, which are the
    # same locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode
    # analysis > Matrix equation formulation; take the cross section through E₂, H₁, H₃.)
    # Therefore, oind2d_ε₂₂ and oind2d_μ₁₁ are evaluated at the same locations, so they
    # should be the same object indices.
    @assert isequal(oind2d′_pₗ, oind2d_pₗ)

    # Smooth the material parameters.
    smooth_param!(εₜarr, tuple(oind2d_pₜ), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # εₜ tensors (rank-0, so scalars)
    smooth_param!(µₜarr, tuple(oind2d_pₜ), shp4oind, μind4oind, μₜ4μind, gtₑ, l, lg, σ, ∆τ, ishₜ˔shp)  # μₜ tensors (rank-0, so scalars)
    smooth_param!(µₗarr, tuple(oind2d_pₗ), shp4oind, μind4oind, μₗ4μind, gtₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(
    β::Number,  # propagation constant
    fₜ::AbstractVector{<:Number},  # calculated transverse field Fₜ as column vector
    ft::FieldType,  # type of input field Fₜ
    ω::Number,
    Ps::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    ∇̽s::NTuple{2,NTuple{2,AbstractMatrix{<:Number}}},
    πcmps::NTuple{2,AbstractMatrix{<:Number}},
    mdl::ModelTE
)
    # Calculate the transverse components of the complementary field.
    βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βf′ₜ4fₜ * fₜ) ./ β

    eₜ, hₜ = ft==EE ? (fₜ,f′ₜ) : (f′ₜ,fₜ)

    # Calculate the longitudinal component of the H-field.
    ⅈhₗ4eₜ = create_ⅈhₗ4eₜ(ω, Ps, ∇̽s)  # matrix
    hₗ = (ⅈhₗ4eₜ * eₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz_grid = size(mdl.grid)
    E₂ = reshape(eₜ, sz_grid)
    H₁ = reshape(hₜ, sz_grid)
    H₃ = reshape(hₗ, sz_grid)

    return (E=E₂, H=(H₁, H₃))
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting_complex(
    E::NTuple{1,AbstractVector{<:Number}},  # (E₂)
    H::NTuple{2,AbstractVector{<:Number}},  # (H₁, H₃)
    mdl::ModelTE
)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    E₂ = E
    H₁, H₃ = H

    dir₁ = 1

    Ftemp = Array{ComplexF64}(undef, size(grid))  # vector; temporary storage

    # Calculate S₁.
    S₁ = Array{ComplexF64}(undef, size(grid))  # vector
    m̂₁H₃ = Ftemp; interp_field!(m̂₁H₃, H₃, HH, dir₁, ∆l, ∆l⁻¹, boundft, isbloch)
    S₁ .= E₂ .* conj.(m̂₁H₃)

    # Calculate S₃.
    S₃ = Array{ComplexF64}(undef, size(grid))  # vector
    S₃₋ = Ftemp; S₃₋ .= E₂ .* conj.(H₁)  # real array located at E₂
    interp_field!(S₃, S₃₋, EE, dir₁, ∆l, ∆l⁻¹, boundft, isbloch, α=-1)

    return S₁, S₃
end
