module MaxwellGuidedMode

using Reexport
@reexport using MaxwellBase
using SimpleConstants
using VoxelwiseConstantMapping
using StaticArrays
using Parameters
using ArgCheck
using Arpack
using Makie

import LinearAlgebra: normalize!

include("model/model.jl")
include("power.jl")
include("matrix.jl")
include("calc_mode.jl")

end
