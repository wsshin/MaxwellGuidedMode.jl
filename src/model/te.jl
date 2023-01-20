# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K²,K₊₂, P,AP} for the TE waveguide mode
# mode equations for 1D cross section.
#
# The type parameter AP specifies device-specific arrays types (e.g., CuArray) and is
# user-defined in the constructor.
#
# Note that the TE mode has (Ey, Hx, Hz) as nonzero components; this is unlike TE Maxwell's
# equations that have two E-field components and one H-field component.

const ModelTE{P,AP} = Model{1,1,2, 1,1, 0,1, 1,3, P,AP}

# Convenience constructor
function ModelTE(grid::Grid; APtype::Type=Array{ComplexF,3})
    ft_eq = EE

    cmpₛ = SInt(1)  # shapes in transverse dimension (x-axis)

    cmpₑₜ = SInt(2)  # transverse E-field (y-component)
    cmpₘₜ = SInt(1)  # transverse H-field (x-component)

    cmpₑₗ = SInt()  # longitudinal E-field (no component)
    cmpₘₗ = SInt(3)  # longitudinal H-field (z-component)

    iseₜ˔shp = true
    ishₜ˔shp = false

    return ModelTE{eltype(APtype),APtype}(; ft_eq, grid, cmpₛ, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
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
    # Because Ey and Hx are at the same locations, smoothing them can use the same object
    # index.
    oind2d_pₜ = create_oind_array(N)  # for εₜ (= εyy) and μₜ (= μxx)
    oind2d_pₗ = create_oind_array(N)  # for μₗ (= μzz)
    oind2d′_pₗ = create_oind_array(N)  # for @assert below

    # Assign the material parameters.
    # Note that μₜarr uses gtₑ rather than gtₘ, because it is rank-0 tensor and its locations
    # are the E-field locations; see the definition MaxwellBase/assign_param!().
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation; take the cross section through Ey, Hx, Hz.)
    assign_param!(εₜarr, tuple(oind2d_pₗ), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # εₜ tensors (rank-0, so scalars)
    assign_param!(µₜarr, tuple(oind2d′_pₗ), shp4oind, μind4oind, μₜ4μind, gtₑ, τlg, isbloch)  # μₜ tensors (rank-0, so scalars)
    assign_param!(µₗarr, tuple(oind2d_pₜ), shp4oind, μind4oind, μₗ4μind, gtₘ, τlg, isbloch)  # μₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  Specifically, oind2d_εyy and oind2d_μxx
    # are evaluated at the locations surrounding the εyy and μxx locations, which are the
    # same locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode
    # analysis > Matrix equation formulation; take the cross section through Ey, Hx, Hz.)
    # Therefore, oind2d_εyy and oind2d_μxx are evaluated at the same locations, so they
    # should be the same object indices.
    @assert isequal(oind2d′_pₗ, oind2d_pₗ)

    # Smooth the material parameters.
    smooth_param!(εₜarr, tuple(oind2d_pₜ), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # εₜ tensors (rank-0, so scalars)
    smooth_param!(µₜarr, tuple(oind2d_pₜ), shp4oind, μind4oind, μₜ4μind, gtₑ, l, lg, σ, ∆τ, ishₜ˔shp)  # μₜ tensors (rank-0, so scalars)
    smooth_param!(µₗarr, tuple(oind2d_pₗ), shp4oind, μind4oind, μₗ4μind, gtₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(β::Number,  # propagation constant
                         fₜ::AbsVecNumber,  # calculated transverse field Fₜ as column vector
                         ft::FieldType,  # type of input field Fₜ
                         ω::Number,
                         Ps::Tuple22{AbsMatNumber},
                         ∇̽s::Tuple22{AbsMatNumber},
                         πcmps::Tuple2{AbsMatNumber},
                         mdl::ModelTE)
    # Calculate the transverse components of the complementary field.
    βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βf′ₜ4fₜ * fₜ) ./ β

    eₜ, hₜ = ft==EE ? (fₜ,f′ₜ) : (f′ₜ,fₜ)

    # Calculate the longitudinal component of the H-field.
    ⅈhₗ4eₜ = create_ⅈhₗ4eₜ(ω, Ps, ∇̽s)
    hₗ = (ⅈhₗ4eₜ * eₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz_grid = size(mdl.grid)
    Ey = reshape(eₜ, sz_grid)
    Hx = reshape(hₜ, sz_grid)
    Hz = reshape(hₗ, sz_grid)

    return (E=Ey, H=(Hx, Hz))
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting(E::AbsVecNumber,  # Ey
                  H::Tuple2{AbsVecNumber},  # (Hx, Hz)
                  mdl::ModelTE)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    Ey = E
    Hx, Hz = H

    nX = 1

    Ftemp = ArrComplexF(undef, size(grid))  # vector; temporary storage

    # Calculate Sx.
    Sx = ArrComplexF(undef, size(grid))  # vector
    m̂xHz = Ftemp; interp_field!(m̂xHz, Hz, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)
    Sx .= Ey .* conj.(m̂xHz)

    # Calculate Sz.
    Sz = ArrComplexF(undef, size(grid))  # vector
    Sz₋ = Ftemp; Sz₋ .= Ey .* conj.(Hx)  # real array located at Ey
    interp_field!(Sz, Sz₋, EE, nX, ∆l, ∆l⁻¹, boundft, isbloch, α=-1.0)

    return Sx, Sz
end
