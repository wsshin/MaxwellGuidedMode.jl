module MaxwellGuidedMode

using Reexport
@reexport using MaxwellBase
using Parameters
using ArgCheck
using AbbreviatedTypes; @define_types_with(Float64)
using SimpleConstants
using VoxelwiseConstantMapping
using Arpack
using Dierckx
using LinearAlgebra: normalize!

include("model/model.jl")
include("calc_mode.jl")

end
