# Alias of Model{K,Kₑ,Kₘ, Kₑₜ,Kₘₜ, Kₑₗ,Kₘₗ, K²,K₊₂, P,AP} for the waveguide mode
# equations for full 2D cross section.
#
# The type parameter AP specifies device-specific arrays types (e.g., CuArray) and
# are user-defined in the constructor.
const ModelFull{P,AP} = Model{2,3,3, 2,2, 1,1, 4,4, P,AP}

# Convenience constructor
function ModelFull(grid::Grid; APtype::Type=Array{ComplexF,4})
    ft_eq = EE

    cmpₑₜ = SInt(1,2)  # transverse E-field (1st and 2nd components)
    cmpₘₜ = SInt(1,2)  # transverse H-field (1st and 2nd components)

    cmpₑₗ = SInt(3)  # longitudinal E-field (3rd component)
    cmpₘₗ = SInt(3)  # longitudinal H-field (3rd component)

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
    oind2d_εxx = create_oind_array(N)
    oind2d_εyy = create_oind_array(N)
    oind2d_εzz = create_oind_array(N)
    oind2d_εoo = create_oind_array(N)

    oind2d_μxx = create_oind_array(N)
    oind2d_μyy = create_oind_array(N)
    oind2d_μzz = create_oind_array(N)
    oind2d_μoo = create_oind_array(N)

    # Assign the material parameters.
    # (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis > Matrix
    # equation formulation.)
    assign_param!(εₜarr, (oind2d_εyy,oind2d_εxx), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # diagonal entries of εₜ tensors
    assign_param!(εₜarr, tuple(oind2d_μzz), shp4oind, εind4oind, εₜ4εind, gtₑ, τlg, isbloch)  # off-diagonal entries of εₜ tensors
    assign_param!(µₗarr, tuple(oind2d_εoo), shp4oind, μind4oind, μₗ4μind, gtₘ, τlg, isbloch)  # μₗ tensors (rank-0, so scalars)

    assign_param!(µₜarr, (oind2d_μyy,oind2d_μxx), shp4oind, μind4oind, μₜ4μind, gtₘ, τlg, isbloch)  # diagonal entries of μₜ tensors
    assign_param!(µₜarr, tuple(oind2d_εzz), shp4oind, μind4oind, μₜ4μind, gtₘ, τlg, isbloch)  # off-diagonal entries of μₜ tensors
    assign_param!(εₗarr, tuple(oind2d_μoo), shp4oind, εind4oind, εₗ4εind, gtₑ, τlg, isbloch)  # εₗ tensors (rank-0, so scalars)

    # Some temporary arrays should be identical.  For example, oind2d_εyy and oind2d_μxx are
    # evaluated at the locations surrounding the εyy and μxx locations, which are the same
    # locations.  (See the figure in L10 - Eigenmode Analysis > 1.3 Waveguide mode analysis
    # > Matrix equation formulation.)  Therefore, oind2d_εyy and oind2d_μxx are evaluated at
    # the same locations, so they should be the same object indices.
    @assert isequal(oind2d_εxx, oind2d_μyy)
    @assert isequal(oind2d_εyy, oind2d_μxx)
    @assert isequal(oind2d_εoo, oind2d_εzz)
    @assert isequal(oind2d_μoo, oind2d_μzz)

    # Smooth the material parameters.
    smooth_param!(εₜarr, (oind2d_εxx,oind2d_εyy), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # diagonal entries of εₜ tensors
    smooth_param!(εₜarr, tuple(oind2d_εoo), shp4oind, εind4oind, εₜ4εind, gtₑ, l, lg, σ, ∆τ, iseₜ˔shp)  # off-diagonal entries of εₜ tensors
    smooth_param!(µₗarr, tuple(oind2d_μzz), shp4oind, μind4oind, μₗ4μind, gtₘ, l, lg, σ, ∆τ, ishₗ˔shp)  # μₗ tensors (rank-0, so scalars)

    smooth_param!(µₜarr, (oind2d_μxx,oind2d_μyy), shp4oind, μind4oind, μₜ4μind, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # diagonal entries of μₜ tensors
    smooth_param!(µₜarr, tuple(oind2d_μoo), shp4oind, μind4oind, μₜ4μind, gtₘ, l, lg, σ, ∆τ, ishₜ˔shp)  # off-diagonal entries of μₜ tensors
    smooth_param!(εₗarr, tuple(oind2d_εzz), shp4oind, εind4oind, εₗ4εind, gtₑ, l, lg, σ, ∆τ, iseₗ˔shp)  # εₗ tensors (rank-0, so scalars)

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
    mdl::ModelFull
)
    ft′ = alter(ft)

    # Calculate the transverse components of the complementary field.
    βf′ₜ4fₜ = create_βf′ₜ4fₜ(ft, ω, Ps, ∇̽s, πcmps)
    f′ₜ = (βf′ₜ4fₜ * fₜ) ./ β

    # Calculate the longitudinal components of the main and complementary fields.
    ⅈf′ₗ4fₜ = create_ⅈf′ₗ4fₜ(ft, ω, Ps, ∇̽s)
    ⅈfₗ4f′ₜ = create_ⅈf′ₗ4fₜ(ft′, ω, Ps, ∇̽s)

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

    Fx = selectdim(Fₜ, d, 1)
    Fy = selectdim(Fₜ, d, 2)

    F′x = selectdim(F′ₜ, d, 1)
    F′y = selectdim(F′ₜ, d, 2)

    sz_grid = size(mdl.grid)
    Fz = reshape(fₗ, sz_grid)
    F′z = reshape(f′ₗ, sz_grid)

    F = (Fx, Fy, Fz)
    F′ = (F′x, F′y, F′z)

    E, H = ft==EE ? (F,F′) : (F′,F)

    return (E=E, H=H)
end

# See the field configurations in L10 - Eigenmode Analysis > Matrix equation formulation.
function poynting(
    E::Tuple3{AbsMatNumber},  # (Ex, Ey, Ez)
    H::Tuple3{AbsMatNumber},  # (Hx, Hy, Hz)
    mdl::ModelFull
)
    boundft = mdl.boundft
    grid = mdl.grid
    isbloch = grid.isbloch

    ∆l, ∆l⁻¹ = create_∆ls(grid, boundft)

    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    nX, nY = 1, 2

    Ftemp = ArrComplexF(undef, size(grid))  # matrix; temporary storage

    # Calculate Sx.
    Sx = ArrComplexF(undef, size(grid))  # matrix

    m̂xHz = Ftemp; interp_field!(m̂xHz, Hz, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hz at Ey-locations
    Sx .= Ey .* conj.(m̂xHz)  # EyHz evaluated at centers of x-normal edges

    m̂xHy = Ftemp; interp_field!(m̂xHy, Hy, HH, nX, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hy at Ez-locations
    m̂xHy .= Ez .* conj.(m̂xHy)  # EzHy evaluated at end points of x-normal edges; need to interpolate at centers
    Sx₋ = m̂xHy

    interp_field!(Sx, Sx₋, EE, nY, ∆l, ∆l⁻¹, boundft, isbloch, α=-1.0, overwrite=false)  # interpolate Sx₋ at centers of x-normal edges

    # Calculate Sy.
    Sy = ArrComplexF(undef, size(grid))  # matrix

    m̂yHz = Ftemp; interp_field!(m̂yHz, Hz, HH, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hz at Ex-locations
    Sy .= -1.0 .* Ex .* conj.(m̂yHz)  # ExHz evaluated at centers of y-normal edges

    m̂yHx = Ftemp; interp_field!(m̂yHx, Hx, HH, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # intepolate Hx at Ez-locations
    m̂yHx .= Ez .* conj.(m̂yHx)  # EzHx evaluated at end points of y-normal edges; need to interpolate at centers
    Sy₊ = m̂yHx

    interp_field!(Sy, Sy₊, EE, nX, ∆l, ∆l⁻¹, boundft, isbloch, overwrite=false)  # interpolate Sy₊ at centers of y-normal edges

    # Calculate Sz.
    Sz = ArrComplexF(undef, size(grid))  # matrix

    Sz₊ = Ftemp; Sz₊ .= Ex .* conj.(Hy)  # Ex and Hy are already co-located
    interp_field!(Sz, Sz₊, EE, nY, ∆l, ∆l⁻¹, boundft, isbloch)  # Sz₊ interpolated at centers of pixels

    Sz₋ = Ftemp; Sz₋ .= Ey .* conj.(Hx)  # Ey and Hx are already co-located
    interp_field!(Sz, Sz₋, EE, nX, ∆l, ∆l⁻¹, boundft, isbloch, α=-1.0, overwrite=false)  # Sz₋ interpolated at centers of pixels

    return Sx, Sy, Sz
end
