using Oceananigans
using NCDatasets
using TOML
using Printf

# Run:
#   julia --project=. forward_model.jl CONFIG.toml PROFILE.ascii OUTPUT.nc [BATHYMETRY] [STOP_TIME]
#
# PROFILE.ascii: two whitespace-separated columns, z [m] and rho [kg/m^3].
# Rows must be ordered from bottom to top and located at z-cell centers.
#
# BATHYMETRY (optional) names a table [bathymetry.<name>] in CONFIG.toml:
#   type = "flat"                                   flat bottom at z = -H
#   type = "gaussian", center, height, width        bottom(x) = -H + height * exp(-(x - center)^2 / (2 width^2))
# Default: flat bottom.
#
# STOP_TIME (optional) overrides run.stop_time; used for short preflight checks.

function read_profile(path, Nz, z_expected, Δz)
    z_read = Float64[]
    rho_read = Float64[]

    for (line_number, line) in enumerate(eachline(path))
        line = strip(first(split(line, '#'; limit=2)))
        isempty(line) && continue

        columns = split(line)
        length(columns) == 2 ||
            error("$path:$line_number: expected exactly two columns: z rho")

        z = tryparse(Float64, columns[1])
        rho = tryparse(Float64, columns[2])
        (z === nothing || rho === nothing) &&
            error("$path:$line_number: could not parse z and rho as numbers")

        push!(z_read, z)
        push!(rho_read, rho)
    end

    length(z_read) == Nz ||
        error("Profile has $(length(z_read)) rows; grid.Nz is $Nz")

    all(isfinite, z_read) || error("Profile contains non-finite z values")
    all(rho -> isfinite(rho) && rho > 0, rho_read) ||
        error("Profile densities must be finite and positive")

    # The profile is specified at cell centers, not at grid faces.
    tolerance = max(1e-8, 1e-5 * Δz)
    maximum(abs.(z_read .- z_expected)) <= tolerance ||
        error("Profile z values must match the grid's bottom-to-top cell centers")

    return rho_read
end

# A localized horizontal acceleration [m/s^2].
# The temporal Gaussian integrates to approximately delta_u.
# cos(pi * (z + H) / H) has zero depth mean: it excites baroclinic motion.
@inline function impulse(x, z, t, p)
    horizontal = exp(-0.5 * ((x - p.x_source) / p.sigma_x)^2)
    vertical = cos(pi * (z + p.H) / p.H)
    temporal = exp(-0.5 * ((t - p.t0) / p.tau)^2) /
               (sqrt(2pi) * p.tau)

    return p.delta_u * horizontal * vertical * temporal
end

# Returns `nothing` for a flat bottom, otherwise a function bottom(x) giving
# the height of the sea floor (z, negative) at horizontal position x.
function bottom_function(spec, Lx, H)
    kind = get(spec, "type", "flat")

    kind == "flat" && return nothing

    if kind == "gaussian"
        center = Float64(spec["center"])
        height = Float64(spec["height"])
        width = Float64(spec["width"])

        0 < center < Lx || error("Seamount center must lie inside (0, Lx)")
        0 < height < H || error("Seamount height must lie inside (0, H)")
        width > 0 || error("Seamount width must be positive")

        return x -> -H + height * exp(-0.5 * ((x - center) / width)^2)
    end

    error("Unknown bathymetry type \"$kind\"; use \"flat\" or \"gaussian\"")
end

# Kinematic nonhydrostatic pressure [m^2/s^2]; the field's location differs between
# Oceananigans versions.
function nonhydrostatic_pressure(model)
    hasproperty(model, :pressures) && hasproperty(model.pressures, :pNHS) &&
        return model.pressures.pNHS
    hasproperty(model, :nonhydrostatic_pressure) &&
        return model.nonhydrostatic_pressure
    error("Cannot find the nonhydrostatic pressure field in this Oceananigans version")
end

function main(args)
    3 <= length(args) <= 5 ||
        error("Usage: julia --project=. forward_model.jl CONFIG.toml PROFILE.ascii OUTPUT.nc [BATHYMETRY] [STOP_TIME]")

    config_path, profile_path, output_path = args[1:3]
    bathymetry_name = length(args) >= 4 ? args[4] : "flat"
    cfg = TOML.parsefile(config_path)

    grid_cfg = cfg["grid"]
    physics_cfg = cfg["physics"]
    impulse_cfg = cfg["impulse"]
    run_cfg = cfg["run"]
    output_cfg = cfg["output"]

    Nx = Int(grid_cfg["Nx"])
    Nz = Int(grid_cfg["Nz"])
    Lx = Float64(grid_cfg["Lx"])
    H = Float64(grid_cfg["H"])

    Nx >= 16 && Nz >= 8 || error("Increase Nx and Nz")
    Lx == 30.0 && H == 2.0 ||
        error("This forward model requires grid.Lx = 30 and grid.H = 2")

    g = Float64(physics_cfg["g"])
    rho_ref = Float64(physics_cfg["rho_ref"])
    nu = Float64(physics_cfg["nu"])
    kappa = Float64(physics_cfg["kappa"])

    g > 0 && rho_ref > 0 && nu >= 0 && kappa >= 0 ||
        error("Require g, rho_ref > 0 and nu, kappa >= 0")

    delta_u = Float64(impulse_cfg["delta_u"])
    sigma_x = Float64(impulse_cfg["sigma_x"])
    offset_from_right = Float64(impulse_cfg["offset_from_right"])
    t0 = Float64(impulse_cfg["t0"])
    tau = Float64(impulse_cfg["tau"])

    sigma_x > 0 && tau > 0 && t0 >= 0 ||
        error("Require sigma_x, tau > 0 and t0 >= 0")
    0 < offset_from_right < Lx ||
        error("offset_from_right must be between 0 and Lx")

    dt = Float64(run_cfg["dt"])
    stop_time = length(args) >= 5 ? parse(Float64, args[5]) : Float64(run_cfg["stop_time"])
    output_interval = Float64(output_cfg["interval"])

    dt > 0 && stop_time > 0 && output_interval > 0 ||
        error("Require dt, stop_time, and output interval > 0")
    t0 < stop_time || error("Impulse center t0 must precede stop_time")
    !isfile(output_path) ||
        error("Refusing to overwrite existing output: $output_path")

    Δx = Lx / Nx
    Δz = H / Nz
    x_centers = [(i - 0.5) * Δx for i in 1:Nx]
    z_centers = [-H + (k - 0.5) * Δz for k in 1:Nz]

    rho_initial = read_profile(profile_path, Nz, z_centers, Δz)

    # Boussinesq buoyancy: b = -g * (rho - rho_ref) / rho_ref.
    b_initial = @. -g * (rho_initial - rho_ref) / rho_ref

    # Bathymetry
    bathymetry_tables = get(cfg, "bathymetry", Dict{String, Any}())
    spec = if haskey(bathymetry_tables, bathymetry_name)
        bathymetry_tables[bathymetry_name]
    elseif bathymetry_name == "flat"
        Dict{String, Any}("type" => "flat")
    else
        error("No [bathymetry.$bathymetry_name] table in $config_path")
    end
    bottom = bottom_function(spec, Lx, H)

    underlying_grid = RectilinearGrid(
        size = (Nx, Nz),
        halo = (4, 4),
        x = (0.0, Lx),
        z = (-H, 0.0),
        topology = (Bounded, Flat, Bounded)
    )

    # With Flat y, the bottom-height function takes x only.
    grid = isnothing(bottom) ? underlying_grid :
           ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    bottom_height = isnothing(bottom) ? fill(-H, Nx) : [bottom(x) for x in x_centers]

    forcing_parameters = (;
        x_source = Lx - offset_from_right,
        sigma_x, t0, tau, delta_u, H
    )
    u_forcing = Forcing(impulse; parameters=forcing_parameters)

    # Ask explicitly for a separate hydrostatic pressure anomaly, so the model's
    # nonhydrostatic pressure is exactly the part NOT given by integrating density
    # (rigid-lid surface pressure + true nonhydrostatic pressure). It is saved
    # below as p_nh; Python adds it to g * integral(rho dz) for the full pressure.
    model_kwargs = (;
        hydrostatic_pressure_anomaly = CenterField(grid),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        advection = WENO(),
        closure = ScalarDiffusivity(; ν=nu, κ=kappa),
        forcing = (; u=u_forcing)
    )

    # With an immersed seamount, the default FFT pressure solver ignores the
    # boundary and lets some flow pass through it. The conjugate-gradient
    # solver enforces no flow through the seamount (slower, more accurate).
    if !isnothing(bottom)
        solver_name = get(run_cfg, "immersed_pressure_solver", "cg")
        if solver_name == "cg"
            isdefined(Oceananigans.Solvers, :ConjugateGradientPoissonSolver) ||
                error("This Oceananigans version has no ConjugateGradientPoissonSolver; " *
                      "update Oceananigans or set run.immersed_pressure_solver = \"fft\"")
            model_kwargs = merge(model_kwargs, (;
                pressure_solver = Oceananigans.Solvers.ConjugateGradientPoissonSolver(grid)))
        elseif solver_name != "fft"
            error("run.immersed_pressure_solver must be \"cg\" or \"fft\"")
        end
    end

    model = NonhydrostaticModel(grid; model_kwargs...)

    # With Flat y, initialization functions take (x, z), not (x, y, z).
    # Map each cell-center z to its corresponding supplied profile value.
    initial_b(x, z) =
        b_initial[clamp(floor(Int, (z + H) / Δz) + 1, 1, Nz)]

    set!(model; u=0.0, b=initial_b)

    simulation = Simulation(model; Δt=dt, stop_time=stop_time)

    # In Julia the variable is (z, x, time); on disk, NetCDF readers
    # such as Python/xarray see rho(time, x, z).
    ds = NCDataset(output_path, "c")

    try
        defDim(ds, "z", Nz)
        defDim(ds, "x", Nx)
        defDim(ds, "t", Inf)

        defVar(ds, "z", z_centers, ("z",);
               attrib=Dict("units" => "m", "positive" => "up"))
        defVar(ds, "x", x_centers, ("x",);
               attrib=Dict("units" => "m"))
        defVar(ds, "bottom_height", bottom_height, ("x",);
               attrib=Dict("units" => "m",
                           "long_name" => "Analytic sea-floor height at x centers"))
        defVar(ds, "t", Float64, ("t",);
               attrib=Dict("units" => "seconds since simulation start"))

        defVar(ds, "rho", Float64, ("z", "x", "t");
               attrib=Dict(
                   "units" => "kg m-3",
                   "long_name" => "Boussinesq-equivalent density"
               ))

        defVar(ds, "p_nh", Float64, ("z", "x", "t");
               attrib=Dict(
                   "units" => "Pa",
                   "long_name" => "Pressure not given by g * integral(rho dz): " *
                                  "rigid-lid surface pressure + nonhydrostatic pressure",
                   "gauge" => "mean over the top row of cells is zero at every time"
               ))

        ds.attrib["rho_ref_kg_m-3"] = rho_ref
        ds.attrib["bathymetry"] = bathymetry_name
        ds.attrib["source"] = "Right-side, zero-depth-mean horizontal impulse"

        frame_number = Ref(0)
        last_saved_time = Ref(-Inf)

        function write_frame!(sim)
            t = Float64(sim.model.clock.time)

            # Some schedules also fire at t=0; never duplicate a frame.
            t <= last_saved_time[] + 1e-10 && return nothing
            frame_number[] += 1

            # Oceananigans' interior field is indexed (x, z) after
            # selecting the single, flat y index.
            b_xz = Array(interior(sim.model.tracers.b, :, 1, :))

            # The NetCDF variable was defined in Julia as (z, x, time).
            rho_zx = permutedims(
                rho_ref .- (rho_ref / g) .* b_xz,
                (2, 1)
            )

            # The pressure solve fixes p only up to a constant; set the mean
            # over the top row of cells to zero (atmospheric reference).
            p_xz = Array(interior(nonhydrostatic_pressure(sim.model), :, 1, :))
            p_xz .-= sum(p_xz[:, end]) / size(p_xz, 1)
            p_nh_zx = permutedims(rho_ref .* p_xz, (2, 1))

            ds["rho"][:, :, frame_number[]] = rho_zx
            ds["p_nh"][:, :, frame_number[]] = p_nh_zx
            ds["t"][frame_number[]] = t
            last_saved_time[] = t

            @printf(stderr, "Saved density frame %d at t = %.3f s\n",
                    frame_number[], t)
            return nothing
        end

        # Save the supplied initial state, including time = 0.
        write_frame!(simulation)

        # Save at the configured interval during integration.
        add_callback!(
            simulation,
            write_frame!,
            TimeInterval(output_interval);
            name = :save_density
        )

        run!(simulation)

        # Also retain the final state if stop_time falls between
        # scheduled output times.
        write_frame!(simulation)

        @printf(stderr, "Wrote %d density frames to %s\n",
                frame_number[], output_path)
    finally
        close(ds)
    end

    return nothing
end

main(ARGS)