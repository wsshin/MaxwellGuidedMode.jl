export set_ωpml!, set_boundft!, set_Npml!, set_kbloch!  # basic setter functions
export create_e⁻ⁱᵏᴸ
export clear_objs!  # plural because clearing both electric and magnetic quantities
export complete_fields

# Do not export Model; qualify it with the package name MaxwellFDFD, because I would have
# similar types in other packages such as MaxwellSALT and MaxwellGuide.
@with_kw mutable struct Model{P<:Number,  # material parameter tensor's element type
                              K,Kₑ,Kₘ,
                              Kₑₜ,Kₘₜ,Kₑₗ,Kₘₗ,
                              K²,
                              K₊₂,AP<:AbsArr{P,K₊₂}}
    # Frequency
    ωpml::Number = 0.0  # can be complex

    # Field type of equation
    ft_eq::FieldType

    # Grid
    grid::Grid{K}
    cmpₛ::SInt{K}  # Cartesian components of shape dimension
    cmpₑₜ::SInt{Kₑₜ}  # Cartesian components of transverse E-field
    cmpₘₜ::SInt{Kₘₜ}  # Cartesian components of transverse H-field
    cmpₑₗ::SInt{Kₑₗ}  # Cartesian components of longitudinal E-field
    cmpₘₗ::SInt{Kₘₗ}  # Cartesian components of longitudinal H-field

    # Boundary properties (in transverse directions)
    boundft::SVec{K,FieldType} = SVec(ntuple(k->EE, Val(K)))
    Npml::Tuple2{SInt{K}} = (SVec(ntuple(k->0, Val(K))), SVec(ntuple(k->0, Val(K))))
    kbloch::SFloat{K} = SVec(ntuple(k->0.0, Val(K)))  # [kx_bloch, ky_bloch, kz_bloch]

    # Material parameter arrays
    εₜarr::AP = create_param_array(P, grid.N, ncmp=Kₑₜ)  # filled with zeros
    μₜarr::AP = create_param_array(P, grid.N, ncmp=Kₘₜ)  # filled with zeros

    εₗarr::AP = create_param_array(P, grid.N, ncmp=Kₑ-Kₑₜ)  # filled with zeros
    μₗarr::AP = create_param_array(P, grid.N, ncmp=Kₘ-Kₘₜ)  # filled with zeros

    # Temporary storages for assignment and smoothing of material parameters
    oind2shp::Vector{Shape{K,K²}} = Shape{K,K²}[]
    oind2εind::Vector{ParamInd} = ParamInd[]
    oind2μind::Vector{ParamInd} = ParamInd[]
    εind2ε::Vector{S²Mat3{P}} = S²Mat3{P}[]
    μind2μ::Vector{S²Mat3{P}} = S²Mat3{P}[]

    # Boolean flags indicating if the transverse E- and H-field dimensions are orthogonal to
    # the shape dimensions.  Used for material parameter smoothing in calc_matparams!().
    iseₜ˔shp::Bool
    ishₜ˔shp::Bool

    # Indexing scheme for DOFs
    order_cmpfirst::Bool = true
end

function Base.size(mdl::Model{P,K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}, ft::FieldType) where {P,K,Kₑ,Kₘ,Kₑₜ,Kₘₜ}
    Kf = ft==EE ? Kₑₜ : Kₘₜ
    sz_grid = size(mdl.grid)
    sz = mdl.order_cmpfirst ? (Kf, sz_grid...) : (sz_grid..., Kf)

    return sz
end

Base.length(mdl::Model, ft::FieldType) = prod(size(mdl, ft))

# Basic setters
set_ωpml!(mdl::Model, ωpml::Number) = (mdl.ωpml = ωpml; nothing)
set_boundft!(mdl::Model{P,K}, boundft::AbsVec{FieldType}) where {P,K} = (mdl.boundft = SVec{K}(boundft); nothing)
set_Npml!(mdl::Model{P,K}, Npml::Tuple2{AbsVecInteger}) where {P,K} = (mdl.Npml = SVec{K}.(Npml); nothing)
set_kbloch!(mdl::Model{P,K}, kbloch::AbsVecReal) where {P,K} = (mdl.kbloch = SVec{K}(kbloch); nothing)

create_e⁻ⁱᵏᴸ(mdl::Model{P,K}) where {P,K} = iszero(mdl.kbloch) ? ones(SFloat{K}) :
                                                             exp.(-im .* mdl.kbloch .* mdl.grid.L)

# Main functions
function clear_objs!(mdl::Model)
    @unpack εₜarr, μₜarr, εₗarr, μₗarr, oind2shp, oind2εind, oind2μind, εind2ε, μind2μ = mdl

    εₜarr .= 0
    μₜarr .= 0

    εₗarr .= 0
    μₗarr .= 0

    empty!(oind2shp)
    empty!(oind2εind)
    empty!(oind2μind)
    empty!(εind2ε)
    empty!(μind2μ)

    return nothing
end

MaxwellBase.add_obj!(mdl::Model{P,K}, matname::String, shapes::AbsVec{<:Shape{K}};
                     ε::AbsMatParam=1.0, μ::AbsMatParam=1.0) where {P,K} =
    add_obj!(mdl, matname, ε=ε, μ=μ, tuple(shapes...))

function MaxwellBase.add_obj!(mdl::Model{P,K}, matname::String, shapes::Shape{K}...;
                              ε::AbsMatParam=1.0, μ::AbsMatParam=1.0) where {P,K}
    mat = Material(matname, ε=ε, μ=μ)
    @argcheck isbidirectional(EE, mat, mdl)  # ε should not couple transverse and longitudinal fields
    @argcheck isbidirectional(HH, mat, mdl)  # μ should not couple transverse and longitudinal fields

    @unpack oind2shp, oind2εind, oind2μind, εind2ε, μind2μ = mdl
    for shp = shapes  # shapes is tuple
        obj = Object(shp, mat)
        add_obj!(oind2shp, (oind2εind,oind2μind), (εind2ε,μind2μ), obj)
    end

    return nothing
end

# Returns true if the material parameter does not couple the transverse and longitudinal
# components of the fields; false otherwise.
function isbidirectional(ft::FieldType, mat::Material, mdl::Model)
    nft = Int(ft)
    @unpack cmpₑₜ, cmpₘₜ, cmpₑₗ, cmpₘₗ = mdl

    cmpₜ = (cmpₑₜ, cmpₘₜ)[nft]
    cmpₗ = (cmpₑₗ, cmpₘₗ)[nft]

    param = mat.param[nft]

    return iszero(param[cmpₜ,cmpₗ]) && iszero(param[cmpₗ,cmpₜ])
end

# Implement the following functions for each specific Model.
function calc_matparams! end
function complete_fields end

function field2vec(E::NTuple{Kₑ,AbsVecNumber{K}}, H::NTuple{Kₘ,AbsVecNumber{K}},
                   mdl::Model{P,K,Kₑ,Kₘ}
                   ) where {P,K,Kₑ,Kₘ}
    @unpack ft_eq, cmpₑₜ, cmpₘₜ, order_cmpfirst = mdl

    U = ft_eq==EE ? E : H
    cmpᵤₜ = ft_eq==EE ? cmpₑₜ : cmpₘₜ
    Uₜ = cat(U[cmpᵤₜ]..., dims=K+1)

    if order_cmpfirst
        Uₜ = permutedims(Uₜ, circshift(1:K, 1))
    end

    return field_arr2vec(Uₜ; order_cmpfirst)
end

include("poynting.jl")
include("calc_mode.jl")

# Specific models
include("full.jl")
include("te.jl")
include("tm.jl")
