using Oceananigans
using NCDatasets
using TOML
using Printf

# Run:
# julia --project=. forward_model.jl config.toml density.ascii
#
# density.ascii: two whitespace-separated columns, z [m] and rho [kg/m^3].
# Rows must be ordered from bottom to top and located at z-cell centers.

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

function main(args)
    length(args) == 3 || error("Usage: julia --project=. forward_model.jl CONFIG.toml PROFILE.ascii OUTPUT.nc")

    config_path, profile_path, output_path = args
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
    stop_time = Float64(run_cfg["stop_time"])
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

    grid = RectilinearGrid(
        size = (Nx, Nz),
        halo = (4, 4),
        x = (0.0, Lx),
        z = (-H, 0.0),
        topology = (Bounded, Flat, Bounded)
    )

    forcing_parameters = (;
        x_source = Lx - offset_from_right,
        sigma_x, t0, tau, delta_u, H
    )
    u_forcing = Forcing(impulse; parameters=forcing_parameters)

    model = NonhydrostaticModel(
        grid;
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        advection = WENO(),
        closure = ScalarDiffusivity(; ν=nu, κ=kappa),
        forcing = (; u=u_forcing)
    )

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
        defVar(ds, "t", Float64, ("t",);
               attrib=Dict("units" => "seconds since simulation start"))

        defVar(ds, "rho", Float64, ("z", "x", "t");
               attrib=Dict(
                   "units" => "kg m-3",
                   "long_name" => "Boussinesq-equivalent density"
               ))

        ds.attrib["rho_ref_kg_m-3"] = rho_ref
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

            ds["rho"][:, :, frame_number[]] = rho_zx
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