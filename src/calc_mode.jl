export calc_mode

function calc_mode(
    mdl::Model, ω::Real,
    mode_prev::NamedTuple{(:ω,:β,:E,:H,:fₜ)};
    nmode::Int=4  # number of modes to calculate
)
    # Even if neff does not change much, β changes from βprev inevitably as ω changes from
    # ωprev following ω = (c / neff) β.  Therefore, it doesn't make sense to assume β ≈
    # βprev.  Instead, assume neff ≈ neff_prev can estimate β accordingly.  See RN -
    # Multilayer Fiber > August 6, 2022.
    #
    # In fact, neff also changes inevitably due to material dispersion in ncore and nclad.
    # Therefore, I think it will be even more reasonable to assume that the normalized index
    # b, instead of neff, is nearly constant for a small change in ω.  However, because
    # calc_mode does not have direct information about ncore and nclad, assuming nearly
    # constant neff, provided that the material dispersion is weak, is the best we can do.
    ωprev, βprev, fₜprev = mode_prev.ω, mode_prev.β, mode_prev.fₜ
    neff_prev = βprev / ωprev  # ω̂ (c₀/L₀) = (c₀/neff) β̂ (1/L₀)
    neff_guess = neff_prev

    βguess = neff_guess * ω

    return calc_mode(mdl, ω, βguess; nmode, fₜref=fₜprev)
end

function calc_mode(
    mdl::Model, ω::Real, βguess::Number;
    nmode::Int=1,  # mode number if fₜref is empty; number of modes to calculate otherwise
    fₜref::AbstractVector{<:Number}=ComplexF[],  # reference field w.r.t. which optimal phase angle is calculated
)
    ## Create the eigenequation and solve it.
    Ps = create_paramops(mdl)
    ∇̽s = create_curls(mdl)
    πcmps = create_πcmps(mdl)

    ft_eq = mdl.ft_eq
    A = create_A(ft_eq, ω, Ps, ∇̽s, πcmps)
    # @show eltype(A)<:Real

    β², f = eigs(A, nev=nmode, sigma=βguess^2)
    eltype(A)<:Real && (β² = real(β²))

    ind_mode = isempty(fₜref) ? nmode : findmax(abs.(fₜref⋅f)[:])[2]

    fₜ = f[:,ind_mode]
    β = .√β²[ind_mode]

    # See RN - NLSE > December 12, 2021 for the explanation of the following procedure.
    #
    # I was wondering if the subsequenty normalization messes up the phase adjusted here.
    # However, the normalization is division by a real number, so it does not change the
    # phase of the fields at all.
    if length(fₜref)==length(fₜ)
        α = fₜref⋅fₜ
        θ = -angle(α)
        fₜ .*= exp(im*θ)
    end

    E, H = complete_fields(β, fₜ, ft_eq, ω, Ps, ∇̽s, πcmps, mdl)
    normalize!(E, H, mdl)

    return (; ω, β, E, H, fₜ)  # not ideal, because transverse components of E or H already contain fₜ
end
