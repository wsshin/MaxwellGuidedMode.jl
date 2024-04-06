# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K²,K₊₂, P,AP} for the TM waveguide mode
# mode equations for 1D cross section.
#
# The type parameter AP specifies device-specific arrays types (e.g., CuArray) and is
# user-defined in the constructor.
#
# Note that the TM mode has (Ex, Ez, Hy) as nonzero components; this is unlike TM Maxwell's
# equations that have two H-field components and one E-field component.

const ModelTM{P,AP} = Model{1,2,1, 1,1, 1,0, 1,3, P,AP}

# Convenience constructor
function ModelTM(grid::Grid; APtype::Type=Array{ComplexF,3})
    ft_eq = HH

    cmpₑₜ = SInt(1)  # transverse E-field (1st component)
    cmpₘₜ = SInt(2)  # transverse H-field (2nd component)

    cmpₑₗ = SInt(3)  # longitudinal E-field (3rd component)
    cmpₘₗ = SInt()  # longitudinal H-field (no component)

    iseₜ˔shp = false
    ishₜ˔shp = true

    return ModelTM{eltype(APtype),APtype}(; ft_eq, grid, cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ, iseₜ˔shp, ishₜ˔shp)
end

# Assign the material parameters on the grid and smooth them.
function calc_matparams!(mdl::ModelTM)
    # Take necessary fields of mdl.
    @unpack grid, boundft,
            εₜarr, εₗarr, µₜarr,
            shp4oind, εind4oind, μind4oind, ε4εind, μ4μind,
            cmpₑₜ, cmpₘₜ, cmpₑₗ,
            iseₜ˔shp, ishₜ˔shp = mdl

    @unpack isbloch, N, l, σ, ghosted = grid
    @unpack lg, τlg, ∆τ = ghosted

    gtₑ = ft2gt.(EE,boundft)
    gtₘ = ft2gt.(HH,boundft)

    εₜ4εind = sub_matprm4pind(ε4εind, cmpₑₜ)
    μₜ4μind = sub_matprm4pind(μ4μind, cmpₘₜ)
    εₗ4εind = sub_matprm4pind(ε4εind, cmpₑₗ)

    iseₗ˔shp = true

    # Create temporary storages.
    # Because Ex and Hy are at the same locations, smoothing them can use the same object
    # index.
    oind2d_pₜ = create_oind_array(N)  # for εₜ (= εxx) and μₜ (= μyy)
    oind2d_pₗ = create_oind_array(N)  # for εₗ (= εzz)
    oind2d′_pₗ = create_oind_array(N)  # for @assert below

    # Assign the material parameters.
    # Note that εₜarr uses gtₘ rather than gtₑ, because it is rank-0 tensor and its locations
    # are the H-field locations; see the definition MaxwellBase/assign_param!().
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation; take the cross section through Ez, Ex, Hy.)
    assign_param!(µₜarr, tuple(oind2d_pₗ), shp4oind, μind4oind, μₜ4μind, gtₘ, τlg, isbloch)  # μₜ tensors (rank-0, so scalars)
    assign_param!(εₜarr, tuple(oind2d′_pₗ), shp4oind, εind4oind, εₜ4εind, gtₘ, τlg, isbloch)  # εₜ tensors (rank-0, so scalars)
    assign_param!(εₗarr, tuple(oind2d_pₜ), shp4oind, εind4oind, εₗ4εind, gtₑ, τlg, isbloch)  # εₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  Specifically, oind2d_εxx and oind2d_μyy
    # are evaluated at the locations surrounding the εxx and μyy locations, which are the
    # same locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode
    # analysis > Matrix equation formulation; take the cross section through Ez, Ex, Hy.)
    # Therefore, oind2d_εxx and oind2d_μyy are evaluated at the same locations, so they
    # should be the same object indices.
    @assert isequal(oind2d′_pₗ, oind2d_pₗ)

    # Smooth the material parameters.
    smooth_param!(µₜarr, tuple(oind2d_pₜ), shp4oind, μind4oind, μₜ4μind, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # μₜ tensors (rank-0, so scalars)
    smooth_param!(εₜarr, tuple(oind2d_pₜ), shp4oind, εind4oind, εₜ4εind, gtₘ, l, lg, σ, ∆τ, iseₜ˔shp)  # εₜ tensors (rank-0, so scalars)
    smooth_param!(εₗarr, tuple(oind2d_pₗ), shp4oind, εind4oind, εₗ4εind, gtₑ, l, lg, σ, ∆τ, iseₗ˔shp)  # εₗ tensors (rank-0, so scalars)

    return nothing
end

function complete_fields(
    β::Number,  # propagation constant
    fₜ::AbsVecNumber,  # calculated transverse field Fₜ as column vector
    ft::FieldType,  # type of input field Fₜ
    ω::Number,
    Ps::Tuple22{AbsMatNumber},
    ∇̽s::Tuple22{AbsMatNumber},
    πcmps::Tuple2{AbsMatNumber},
    mdl::ModelTM
)
    # Calculate the transverse components of the complementary field.
    βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βf′ₜ4fₜ * fₜ) ./ β

    eₜ, hₜ = ft==EE ? (fₜ,f′ₜ) : (f′ₜ,fₜ)

    # Calculate the longitudinal component of the E-field.
    ⅈeₗ4hₜ = create_ⅈeₗ4hₜ(ω, Ps, ∇̽s)
    eₗ = (ⅈeₗ4hₜ * hₜ) ./ im

    # Take the Cartesian components of the fields and reshape them into arrays.
    sz_grid = size(mdl.grid)
    Ex = reshape(eₜ, sz_grid)
    Ez = reshape(eₗ, sz_grid)
    Hy = reshape(hₜ, sz_grid)

    return (E=(Ex, Ez), H=Hy)
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting(
    E::Tuple2{AbsVecNumber},  # (Ex, Ez)
    H::AbsVecNumber,  # Hy
    mdl::ModelTM
)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    Ex, Ez = E
    Hy = H

    nX = 1

    # Calculate Sx.
    m̂xHy = ArrComplexF(undef, size(grid))  # vector
    interp_field!(m̂xHy, Hy, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)
    m̂xHy .= -1.0 .* Ez .* conj.(m̂xHy)
    Sx = m̂xHy

    # Calculate Sz.
    Sz = Ex .* conj.(Hy)  # real array located at Ex

    return Sx, Sz
end
