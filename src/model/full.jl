# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K²,K₊₂, P,AP} for the waveguide mode
# equations for full 2D cross section.
#
# The type parameter AP specifies device-specific arrays types (e.g., CuArray) and
# are user-defined in the constructor.
const ModelFull{P,AP} = Model{2,3,3, 2,2, 1,1, 4,4, P,AP}

# Convenience constructor
function ModelFull(grid::Grid; APtype::Type=Array{ComplexF,4})
    ft_eq = EE

    cmpₑₜ = SVector(1,2)  # transverse E-field (1st and 2nd components)
    cmpₘₜ = SVector(1,2)  # transverse H-field (1st and 2nd components)

    cmpₑₗ = SVector(3)  # longitudinal E-field (3rd component)
    cmpₘₗ = SVector(3)  # longitudinal H-field (3rd component)

    iseₜ˔shp = false
    ishₜ˔shp = false

    return ModelFull{eltype(APtype),APtype}(; ft_eq, grid, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelFull)
    # Take necessary fields of mdl.
    @unpack grid, boundft,
            εₜarr, µₜarr, εₗarr, µₗarr,
            shp4oind, εind4oind, μind4oind, ε4εind, μ4μind,
            cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ,
            iseₜ˔shp, ishₜ˔shp = mdl

    @unpack isbloch, N, l, σ, ghosted = grid
    @unpack lg, τlg, ∆τ = ghosted

    gtₑ = ft2gt.(EE,boundft)
    gtₘ = ft2gt.(HH,boundft)

    εₜ4εind = sub_matprm4pind(ε4εind, cmpₑₜ)
    εₗ4εind = sub_matprm4pind(ε4εind, cmpₑₗ)
    μₜ4μind = sub_matprm4pind(μ4μind, cmpₘₜ)
    μₗ4μind = sub_matprm4pind(μ4μind, cmpₘₗ)

    iseₗ˔shp = true
    ishₗ˔shp = true

    # Create temporary storages.
    oind2d_ε₁₁ = create_oind_array(N)
    oind2d_ε₂₂ = create_oind_array(N)
    oind2d_ε₃₃ = create_oind_array(N)
    oind2d_ε₀₀ = create_oind_array(N)

    oind2d_μ₁₁ = create_oind_array(N)
    oind2d_μ₂₂ = create_oind_array(N)
    oind2d_μ₃₃ = create_oind_array(N)
    oind2d_μ₀₀ = create_oind_array(N)

    # Assign the material parameters.
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation.)
    assign_param!(εₜarr, (oind2d_ε₂₂,oind2d_ε₁₁), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # diagonal entries of εₜ tensors
    assign_param!(εₜarr, tuple(oind2d_μ₃₃), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # off-diagonal entries of εₜ tensors
    assign_param!(µₗarr, tuple(oind2d_ε₀₀), shp4oind, μind4oind, μₗ4μind, gtₘ, τlg, isbloch)  # μₗ tensors (rank-0, so scalars)

    assign_param!(µₜarr, (oind2d_μ₂₂,oind2d_μ₁₁), shp4oind, μind4oind, μₜ4μind, gtₘ, τlg, isbloch)  # diagonal entries of μₜ tensors
    assign_param!(µₜarr, tuple(oind2d_ε₃₃), shp4oind, μind4oind, μₜ4μind, gtₘ, τlg, isbloch)  # off-diagonal entries of μₜ tensors
    assign_param!(εₗarr, tuple(oind2d_μ₀₀), shp4oind, εind4oind, εₗ4εind, gtₑ, τlg, isbloch)  # εₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  For example, oind2d_ε₂₂ and oind2d_μ₁₁ are
    # evaluated at the locations surrounding the ε₂₂ and μ₁₁ locations, which are the same
    # locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis
    # > Matrix equation formulation.)  Therefore, oind2d_ε₂₂ and oind2d_μ₁₁ are evaluated at
    # the same locations, so they should be the same object indices.
    @assert isequal(oind2d_ε₁₁, oind2d_μ₂₂)
    @assert isequal(oind2d_ε₂₂, oind2d_μ₁₁)
    @assert isequal(oind2d_ε₀₀, oind2d_ε₃₃)
    @assert isequal(oind2d_μ₀₀, oind2d_μ₃₃)

    # Smooth the material parameters.
    smooth_param!(εₜarr, (oind2d_ε₁₁,oind2d_ε₂₂), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # diagonal entries of εₜ tensors
    smooth_param!(εₜarr, tuple(oind2d_ε₀₀), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # off-diagonal entries of εₜ tensors
    smooth_param!(µₗarr, tuple(oind2d_μ₃₃), shp4oind, μind4oind, μₗ4μind, gtₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    smooth_param!(µₜarr, (oind2d_μ₁₁,oind2d_μ₂₂), shp4oind, μind4oind, μₜ4μind, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # diagonal entries of μₜ tensors
    smooth_param!(µₜarr, tuple(oind2d_μ₀₀), shp4oind, μind4oind, μₜ4μind, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # off-diagonal entries of μₜ tensors
    smooth_param!(εₗarr, tuple(oind2d_ε₃₃), shp4oind, εind4oind, εₗ4εind, gtₑ, l, lg, σ, ∆τ, iseₗ˔shp)  # εₗ tensors (rank-0, so scalars)

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
    mdl::ModelFull
)
    ft′ = alter(ft)

    # Calculate the transverse components of the complementary field.
    βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βf′ₜ4fₜ * fₜ) ./ β

    # Calculate the longitudinal components of the main and complementary fields.
    ⅈf′ₗ4fₜ = create_ⅈf′ₗ4fₜ(ft, ω, Ps, ∇̽s)  # matrix
    ⅈfₗ4f′ₜ = create_ⅈf′ₗ4fₜ(ft′, ω, Ps, ∇̽s)  # matrix

    fₗ = (ⅈfₗ4f′ₜ * f′ₜ) ./ im
    f′ₗ = (ⅈf′ₗ4fₜ * fₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz = size(mdl, ft)
    sz′ = size(mdl, ft′)
    @assert sz′ == sz

    Fₜ = reshape(fₜ, sz)
    F′ₜ = reshape(f′ₜ, sz)

    cmpfirst = mdl.cmpfirst
    d = cmpfirst ? 1 : length(sz)

    F₁ = selectdim(Fₜ, d, 1)
    F₂ = selectdim(Fₜ, d, 2)

    F′₁ = selectdim(F′ₜ, d, 1)
    F′₂ = selectdim(F′ₜ, d, 2)

    sz_grid = size(mdl.grid)
    F₃ = reshape(fₗ, sz_grid)
    F′₃ = reshape(f′ₗ, sz_grid)

    F = (F₁, F₂, F₃)
    F′ = (F′₁, F′₂, F′₃)

    E, H = ft==EE ? (F,F′) : (F′,F)

    return (E=E, H=H)
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting_complex(
    E::NTuple{3,AbstractMatrix{<:Number}},  # (E₁, E₂, E₃)
    H::NTuple{3,AbstractMatrix{<:Number}},  # (H₁, H₂, H₃)
    mdl::ModelFull
)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    E₁, E₂, E₃ = E
    H₁, H₂, H₃ = H

    dir₁, dir₂ = 1, 2

    Ftemp = Array{ComplexF64}(undef, size(grid))  # matrix; temporary storage

    # Below, note that the grid is 2D Yee's grid in the plane of the directions 1 and 2.
    # The field components in the direction 3 are defined on this plane, and their locations
    # are the projection of the direction-3 components of 3D Yee's grid onto the plane.

    # Calculate S₁.
    S₁ = Array{ComplexF64}(undef, size(grid))  # matrix

    m̂₁H₃ = Ftemp; interp_field!(m̂₁H₃, H₃, HH, dir₁, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate H₃ at E₂-locations
    S₁ .= E₂ .* conj.(m̂₁H₃)  # E₂H₃ evaluated at centers of edges normal to direction 1

    m̂₁H₂ = Ftemp; interp_field!(m̂₁H₂, H₂, HH, dir₁, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate H₂ at E₃-locations
    m̂₁H₂ .= E₃ .* conj.(m̂₁H₂)  # E₃H₂ evaluated at end points of edges normal to direction 1; need to be interpolated at edge centers
    S₁₋ = m̂₁H₂

    interp_field!(S₁, S₁₋, EE, dir₂, ∆l, ∆l⁻¹, boundft, isbloch, α=-1, overwrite=false)  # interpolate S₁₋ at centers of edges normal to direction 1

    # Calculate S₂.
    S₂ = Array{ComplexF64}(undef, size(grid))  # matrix

    m̂₂H₃ = Ftemp; interp_field!(m̂₂H₃, H₃, HH, dir₂, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate H₃ at E₁-locations
    S₂ .= -1 .* E₁ .* conj.(m̂₂H₃)  # E₁H₃ evaluated at centers of edges normal to direction 2

    m̂₂H₁ = Ftemp; interp_field!(m̂₂H₁, H₁, HH, dir₂, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate H₁ at E₃-locations
    m̂₂H₁ .= E₃ .* conj.(m̂₂H₁)  # E₃H₁ evaluated at end points of edges normal to direction 2; need to be interpolated at edge centers
    S₂₊ = m̂₂H₁

    interp_field!(S₂, S₂₊, EE, dir₁, ∆l, ∆l⁻¹, boundft, isbloch, overwrite=false)  # interpolate S₂₊ at centers of edges normal to direction 2

    # Calculate S₃.
    S₃ = Array{ComplexF64}(undef, size(grid))  # matrix

    S₃₊ = Ftemp; S₃₊ .= E₁ .* conj.(H₂)  # E₁ and H₂ are already co-located
    interp_field!(S₃, S₃₊, EE, dir₂, ∆l, ∆l⁻¹, boundft, isbloch)  # S₃₊ interpolated at centers of pixels

    S₃₋ = Ftemp; S₃₋ .= E₂ .* conj.(H₁)  # E₂ and H₁ are already co-located
    interp_field!(S₃, S₃₋, EE, dir₁, ∆l, ∆l⁻¹, boundft, isbloch, α=-1, overwrite=false)  # S₃₋ interpolated at centers of pixels

    return S₁, S₂, S₃
end
