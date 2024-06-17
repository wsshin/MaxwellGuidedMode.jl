export set_ωpml!, set_boundft!, set_Npml!, set_kbloch!  # basic setter functions
export create_e⁻ⁱᵏᴸ
export clear_objs!  # plural because clearing both electric and magnetic quantities
export complete_fields

# Implement the following functions for each specific Model.
function calc_matparams! end
function complete_fields end
function poynting_complex end

# Do not export Model; qualify it with the package name MaxwellFDFD, because I would have
# similar types in other packages such as MaxwellSALT and MaxwellGuide.
@with_kw mutable struct Model{
    K,Kₑ,Kₘ,
    Kₑₜ,Kₘₜ,Kₑₗ,Kₘₗ,
    K²,K₊₂,
    P<:Number,AP<:AbstractArray{P,K₊₂}  # P: material parameter tensor's element type
}
    # Frequency
    ωpml::Number = 0.0  # can be complex

    # Field type of equation
    ft_eq::AbstractFieldType

    # Grid
    grid::Grid{K}

    # Component information used for creating generalized curls
    # See L10 - Eigenmode Analysis > [Aug/27/2021] Attempt to recover similarity between 1D and 2D cases.
    cmpₑₜ::SVector{Kₑₜ,Int}  # Cartesian components of transverse E-field
    cmpₘₜ::SVector{Kₘₜ,Int}  # Cartesian components of transverse H-field
    cmpₑₗ::SVector{Kₑₗ,Int}  # Cartesian components of longitudinal E-field
    cmpₘₗ::SVector{Kₘₗ,Int}  # Cartesian components of longitudinal H-field

    # Boundary properties (in transverse directions)
    boundft::SVector{K,AbstractFieldType} = fill(Efield(), SVector{K})
    Npml::NTuple{2,SVector{K,Int}} = (zeros(SVector{K,Int}), zeros(SVector{K,Int}))
    kbloch::SVector{K,Float64} = zeros(SVector{K,Float64})  # [kx_bloch, ky_bloch, kz_bloch]

    # Material parameter arrays
    εₜarr::AP = create_param_array(P, grid.N, ncmp=Kₑₜ)  # filled with zeros
    μₜarr::AP = create_param_array(P, grid.N, ncmp=Kₘₜ)  # filled with zeros

    εₗarr::AP = create_param_array(P, grid.N, ncmp=Kₑ-Kₑₜ)  # filled with zeros
    μₗarr::AP = create_param_array(P, grid.N, ncmp=Kₘ-Kₘₜ)  # filled with zeros

    # Temporary storages for assignment and smoothing of material parameters
    shp4oind::Vector{Shape{K,K²}} = Shape{K,K²}[]
    εind4oind::Vector{ParamInd} = ParamInd[]
    μind4oind::Vector{ParamInd} = ParamInd[]
    ε4εind::Vector{SMatrix{3,3,P,9}} = SMatrix{3,3,P,9}[]
    μ4μind::Vector{SMatrix{3,3,P,9}} = SMatrix{3,3,P,9}[]

    # Boolean flags indicating if the transverse E- and H-field dimensions are orthogonal to
    # the shape dimensions.  Used for material parameter smoothing in calc_matparams!().
    iseₜ˔shp::Bool
    ishₜ˔shp::Bool

    # Indexing scheme for DOFs
    cmpfirst::Bool = true
end

function Base.size(mdl::Model{K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}, ft::AbstractFieldType) where {K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}
    Kf = ft==Efield() ? Kₑₜ : Kₘₜ
    sz_grid = size(mdl.grid)
    sz = mdl.cmpfirst ? (Kf, sz_grid...) : (sz_grid..., Kf)

    return sz
end

Base.length(mdl::Model, ft::AbstractFieldType) = prod(size(mdl, ft))

# Basic setters
set_ωpml!(mdl::Model, ωpml::Number) = (mdl.ωpml = ωpml; nothing)
set_boundft!(mdl::Model{K}, boundft::AbstractVector{AbstractFieldType}) where {K} = (mdl.boundft = SVector{K}(boundft); nothing)
set_Npml!(mdl::Model{K}, Npml::NTuple{2,AbstractVector{<:Integer}}) where {K} = (mdl.Npml = SVector{K}.(Npml); nothing)
set_kbloch!(mdl::Model{K}, kbloch::AbstractVector{<:Real}) where {K} = (mdl.kbloch = SVector{K}(kbloch); nothing)

create_e⁻ⁱᵏᴸ(mdl::Model{K}) where {K} =
    iszero(mdl.kbloch) ?
        ones(SVector{K,Float64}) :
        exp.(-im .* mdl.kbloch .* mdl.grid.L)

# Main functions
function clear_objs!(mdl::Model)
    @unpack εₜarr, μₜarr, εₗarr, μₗarr, shp4oind, εind4oind, μind4oind, ε4εind, μ4μind = mdl

    εₜarr .= 0
    μₜarr .= 0

    εₗarr .= 0
    μₗarr .= 0

    empty!(shp4oind)
    empty!(εind4oind)
    empty!(μind4oind)
    empty!(ε4εind)
    empty!(μ4μind)

    return nothing
end

MaxwellBase.add_obj!(
    mdl::Model{K}, matname::String, shapes::AbstractVector{<:Shape{K}};
    ε::AbstractMatParam=1.0, μ::AbstractMatParam=1.0
) where {K} =
    add_obj!(mdl, matname, ε=ε, μ=μ, tuple(shapes...))

function MaxwellBase.add_obj!(
    mdl::Model{K}, matname::String, shapes::Shape{K}...;
    ε::AbstractMatParam=1.0, μ::AbstractMatParam=1.0
) where {K}
    mat = Material(matname, ε=ε, μ=μ)
    @argcheck isbidirectional(Efield(), mat, mdl)  # ε should not couple transverse and longitudinal fields
    @argcheck isbidirectional(Hfield(), mat, mdl)  # μ should not couple transverse and longitudinal fields

    @unpack shp4oind, εind4oind, μind4oind, ε4εind, μ4μind = mdl
    for shp = shapes  # shapes is tuple
        obj = Object(shp, mat)
        add_obj!(shp4oind, (εind4oind,μind4oind), (ε4εind,μ4μind), obj)
    end

    return nothing
end

# Returns true if the material parameter does not couple the transverse and longitudinal
# components of the fields; false otherwise.
function isbidirectional(ft::AbstractFieldType, mat::Material, mdl::Model)
    @unpack cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ = mdl

    cmpₜ = (cmpₑₜ, cmpₘₜ)[ft]
    cmpₗ = (cmpₑₗ, cmpₘₗ)[ft]

    param = mat.param[ft]

    return iszero(param[cmpₜ,cmpₗ]) && iszero(param[cmpₗ,cmpₜ])
end

# This function is not used currently; not sure if this will be needed.
function field2vec(
    E::NTuple{Kₑ,AbstractArray{<:Number,K}},
    H::NTuple{Kₘ,AbstractArray{<:Number,K}},
    mdl::Model{K,Kₑ,Kₘ}
) where {K,Kₑ,Kₘ}
    @unpack ft_eq, cmpₑₜ, cmpₘₜ, cmpfirst = mdl

    U = ft_eq==Efield() ? E : H
    cmpᵤₜ = ft_eq==Efield() ? cmpₑₜ : cmpₘₜ
    Uₜ = cat(U[cmpᵤₜ]..., dims=K+1)

    if cmpfirst
        Uₜ = permutedims(Uₜ, circshift(1:K+1, 1))
    end

    return field_arr2vec(Uₜ; cmpfirst)
end

# Specific models
include("full.jl")
include("te.jl")
include("tm.jl")
