using Oceananigans
using NCDatasets
using TOML
using Printf

# Towed-seamount forward model: a stratified, closed 2D tank (walls at both ends, rigid lid)
# with a 2D ridge ("seamount") towed along the floor at constant speed.
#
# Run:
#   julia --project=. forward_model.jl CONFIG.toml PROFILE.ascii OUTPUT.nc [GEOMETRY] [STOP_TIME]
#
# PROFILE.ascii: two whitespace-separated columns, z [m] and rho [kg/m^3], bottom to top,
# at the grid's z-cell centres. GEOMETRY is accepted for compatibility and ignored.
# STOP_TIME overrides run.stop_time (used for short preflight checks).
#
# The moving seamount ([seamount] in CONFIG.toml):
#   shape      raised cosine, height * cos^2(pi * xi / width) for |xi| < width / 2, zero outside,
#              where xi = x - x_s(t) and x_s(t) is the centre position;
#   motion     starts at rest at start_x and moves toward -x with speed
#              speed * sin^2(pi * (t - t_start) / duration) for t_start <= t <= t_start + duration,
#              zero before and after: a smooth pulse peaking at `speed` halfway through, with total
#              travel speed * duration / 2.
# Oceananigans cannot move an immersed boundary, so the seamount is a moving solid region
# represented by volume penalization: inside a smoothed mask of its shape, the velocity is
# relaxed to the seamount's velocity (u = -speed * ramp, w = 0) and buoyancy to its undisturbed
# value. The underlying grid has no immersed boundary, so the standard FFT pressure solver is used.
#
# Boundaries: free-slip, impenetrable walls at x = 0 and x = Lx, free-slip floor, rigid lid.

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

    tolerance = max(1e-8, 1e-5 * Δz)
    maximum(abs.(z_read .- z_expected)) <= tolerance ||
        error("Profile z values must match the grid's bottom-to-top cell centers")

    return rho_read
end

# ---------------------------------------------------------------- seamount motion and shape

# Speed pulse (fraction of the peak speed, 0 outside the motion window) and the centre
# position (exact integral of the speed).
@inline function tow_ramp(t, p)
    s = clamp(t - p.t_start, 0.0, p.duration)
    return sin(pi * s / p.duration)^2
end
@inline function tow_position(t, p)
    s = clamp(t - p.t_start, 0.0, p.duration)
    return p.start_x - p.speed * (s / 2 - p.duration / (4pi) * sin(2pi * s / p.duration))
end

# Seamount height above the floor and its slope, as a function of xi = x - x_s
@inline function ridge_height(xi, p)
    abs(xi) < p.width / 2 || return 0.0
    return p.height * cos(pi * xi / p.width)^2
end

@inline function ridge_slope(xi, p)
    abs(xi) < p.width / 2 || return 0.0
    return -p.height * (pi / p.width) * sin(2pi * xi / p.width)
end

# Smoothed solid fraction in [0, 1]: ~1 inside the seamount, ~0 in the water. The transition
# is smoothed over `delta` in the direction normal to the surface, and faded out where the
# ridge is thinner than 2 delta, so the bare floor is never penalized.
@inline function solid_mask(x, z, t, p)
    xi = x - tow_position(t, p)
    h = ridge_height(xi, p)
    h > 0 || return 0.0
    distance = (z - (-p.H + h)) / sqrt(1 + ridge_slope(xi, p)^2)   # > 0 above the surface
    return 0.5 * (1 - tanh(distance / p.delta)) * min(1.0, h / (2 * p.delta))
end

# Penalization forcings: relax toward the seamount's motion inside the mask
@inline u_penalty(x, z, t, u, p) = -p.rate * solid_mask(x, z, t, p) * (u + p.speed * tow_ramp(t, p))
@inline w_penalty(x, z, t, w, p) = -p.rate * solid_mask(x, z, t, p) * w
@inline function b_penalty(x, z, t, b, p)
    k = clamp(floor(Int, (z + p.H) / p.Δz) + 1, 1, length(p.b0))
    return -p.rate * solid_mask(x, z, t, p) * (b - p.b0[k])
end

# Kinematic nonhydrostatic pressure [m^2/s^2]; its location differs between Oceananigans versions.
function nonhydrostatic_pressure(model)
    hasproperty(model, :pressures) && hasproperty(model.pressures, :pNHS) &&
        return model.pressures.pNHS
    hasproperty(model, :nonhydrostatic_pressure) &&
        return model.nonhydrostatic_pressure
    error("Cannot find the nonhydrostatic pressure field in this Oceananigans version")
end

function main(args)
    3 <= length(args) <= 5 ||
        error("Usage: julia --project=. forward_model.jl CONFIG.toml PROFILE.ascii OUTPUT.nc [GEOMETRY] [STOP_TIME]")

    config_path, profile_path, output_path = args[1:3]
    cfg = TOML.parsefile(config_path)

    # FFTW threads: all Julia threads by default; set parallel.fftw_threads = 1 if FFTW's
    # multithreading crashes (seen on Windows: EXCEPTION_ACCESS_VIOLATION in fftw_threads_set_callback).
    fftw_threads = Int(get(get(cfg, "parallel", Dict{String, Any}()), "fftw_threads", Threads.nthreads()))
    fftw_threads >= 1 || error("parallel.fftw_threads must be at least 1")
    if isdefined(Oceananigans.Solvers, :FFTW)
        Oceananigans.Solvers.FFTW.set_num_threads(fftw_threads)
    end

    grid_cfg, physics_cfg, seamount_cfg = cfg["grid"], cfg["physics"], cfg["seamount"]
    run_cfg, output_cfg = cfg["run"], cfg["output"]

    Nx, Nz = Int(grid_cfg["Nx"]), Int(grid_cfg["Nz"])
    Lx, H = Float64(grid_cfg["Lx"]), Float64(grid_cfg["H"])
    Nx >= 16 && Nz >= 8 || error("Increase Nx and Nz")
    Lx > 0 && H > 0 || error("Require Lx, H > 0")

    g = Float64(physics_cfg["g"])
    rho_ref = Float64(physics_cfg["rho_ref"])
    nu = Float64(physics_cfg["nu"])
    kappa = Float64(physics_cfg["kappa"])
    g > 0 && rho_ref > 0 && nu >= 0 && kappa >= 0 ||
        error("Require g, rho_ref > 0 and nu, kappa >= 0")

    height = Float64(seamount_cfg["height"])
    width = Float64(seamount_cfg["width"])
    start_x = Float64(seamount_cfg["start_x"])
    speed = Float64(seamount_cfg["speed"])
    t_start = Float64(seamount_cfg["t_start"])
    duration = Float64(seamount_cfg["duration"])
    0 < height < H || error("seamount.height must lie inside (0, H)")
    width > 0 && duration > 0 && t_start >= 0 && speed >= 0 ||
        error("Require seamount.width, duration > 0, t_start >= 0, speed >= 0")

    dt = Float64(run_cfg["dt"])
    stop_time = length(args) >= 5 ? parse(Float64, args[5]) : Float64(run_cfg["stop_time"])
    output_interval = Float64(output_cfg["interval"])
    dt > 0 && stop_time > 0 && output_interval > 0 ||
        error("Require dt, stop_time, and output interval > 0")
    !isfile(output_path) || error("Refusing to overwrite existing output: $output_path")

    Δx, Δz = Lx / Nx, H / Nz
    x_centers = [(i - 0.5) * Δx for i in 1:Nx]
    z_centers = [-H + (k - 0.5) * Δz for k in 1:Nz]

    # Penalization: explicit, so rate * dt must stay well below 1 (default 0.5 / dt).
    rate = Float64(get(run_cfg, "penalization_rate", 0.5 / dt))
    0 < rate * dt <= 1 || error("run.penalization_rate * dt must lie in (0, 1]")
    delta = Float64(get(run_cfg, "mask_smoothing", Δx / 2))
    delta > 0 || error("run.mask_smoothing must be positive")

    # The seamount must stay clear of both walls for the whole run
    p_motion = (; start_x, speed, t_start, duration)
    x_end = tow_position(stop_time, p_motion)
    width / 2 + Δx < min(start_x, x_end) && max(start_x, x_end) < Lx - width / 2 - Δx ||
        error("The seamount would touch a wall: centre moves from $start_x to $(round(x_end, digits=3)) m")

    rho_initial = read_profile(profile_path, Nz, z_centers, Δz)
    b_initial = @. -g * (rho_initial - rho_ref) / rho_ref   # Boussinesq buoyancy

    grid = RectilinearGrid(
        size = (Nx, Nz),
        halo = (4, 4),
        x = (0.0, Lx),
        z = (-H, 0.0),
        topology = (Bounded, Flat, Bounded)
    )

    p = (; start_x, speed, t_start, duration, height, width, H, delta, rate, Δz, b0 = b_initial)
    forcing = (;
        u = Forcing(u_penalty; field_dependencies = :u, parameters = p),
        w = Forcing(w_penalty; field_dependencies = :w, parameters = p),
        b = Forcing(b_penalty; field_dependencies = :b, parameters = p),
    )

    # A separate hydrostatic pressure anomaly makes the model's nonhydrostatic pressure exactly
    # the part not given by integrating density; it is saved as p_nh and added in Python.
    model = NonhydrostaticModel(grid;
        hydrostatic_pressure_anomaly = CenterField(grid),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        advection = WENO(),
        closure = ScalarDiffusivity(; ν=nu, κ=kappa),
        forcing
    )

    initial_b(x, z) = b_initial[clamp(floor(Int, (z + H) / Δz) + 1, 1, Nz)]
    set!(model; u=0.0, b=initial_b)

    simulation = Simulation(model; Δt=dt, stop_time=stop_time)

    # Safety: stop if the time step is too large for the flow (advective CFL)
    max_cfl = Float64(get(run_cfg, "max_cfl", 0.8))
    cfl = AdvectiveCFL(dt)
    function check_cfl(sim)
        c = cfl(sim.model)
        c <= max_cfl || error(@sprintf("Advective CFL %.2f exceeds run.max_cfl = %.2f at t = %.1f s; reduce run.dt",
                                       c, max_cfl, sim.model.clock.time))
        return nothing
    end
    add_callback!(simulation, check_cfl, IterationInterval(50); name = :cfl_check)

    ds = NCDataset(output_path, "c")
    try
        defDim(ds, "z", Nz)
        defDim(ds, "x", Nx)
        defDim(ds, "t", Inf)

        defVar(ds, "z", z_centers, ("z",); attrib=Dict("units" => "m", "positive" => "up"))
        defVar(ds, "x", x_centers, ("x",); attrib=Dict("units" => "m"))
        defVar(ds, "t", Float64, ("t",); attrib=Dict("units" => "seconds since simulation start"))
        defVar(ds, "seamount_x", Float64, ("t",);
               attrib=Dict("units" => "m", "long_name" => "Seamount centre position"))
        defVar(ds, "rho", Float64, ("z", "x", "t");
               attrib=Dict("units" => "kg m-3", "long_name" => "Boussinesq-equivalent density"))
        defVar(ds, "p_nh", Float64, ("z", "x", "t");
               attrib=Dict("units" => "Pa",
                           "long_name" => "Pressure not given by g * integral(rho dz from z to 0): " *
                                          "rigid-lid surface pressure + nonhydrostatic pressure",
                           "gauge" => "mean over the top row of cells is zero at every time"))

        ds.attrib["rho_ref_kg_m-3"] = rho_ref
        ds.attrib["source"] = "Towed raised-cosine seamount, height $height m, width $width m, " *
                                 "sin^2 speed pulse peaking at $speed m/s over $(duration) s from t = $(t_start) s"
        ds.attrib["top_boundary"] = "rigid lid"

        frame_number = Ref(0)
        last_saved_time = Ref(-Inf)

        function write_frame!(sim)
            t = Float64(sim.model.clock.time)
            t <= last_saved_time[] + 1e-10 && return nothing
            frame_number[] += 1

            b_xz = Array(interior(sim.model.tracers.b, :, 1, :))
            rho_zx = permutedims(rho_ref .- (rho_ref / g) .* b_xz, (2, 1))

            # Rigid lid: pressure is fixed only up to a constant; zero mean over the top row.
            p_xz = Array(interior(nonhydrostatic_pressure(sim.model), :, 1, :))
            p_xz .-= sum(p_xz[:, end]) / size(p_xz, 1)
            p_nh_zx = permutedims(rho_ref .* p_xz, (2, 1))

            ds["rho"][:, :, frame_number[]] = rho_zx
            ds["p_nh"][:, :, frame_number[]] = p_nh_zx
            ds["seamount_x"][frame_number[]] = tow_position(t, p_motion)
            ds["t"][frame_number[]] = t
            last_saved_time[] = t

            @printf(stderr, "Saved frame %d at t = %.3f s (seamount at x = %.3f m)\n",
                    frame_number[], t, tow_position(t, p_motion))
            return nothing
        end

        write_frame!(simulation)
        add_callback!(simulation, write_frame!, TimeInterval(output_interval); name = :save_frames)
        run!(simulation)
        write_frame!(simulation)

        @printf(stderr, "Wrote %d frames to %s\n", frame_number[], output_path)
    finally
        close(ds)
    end

    return nothing
end

main(ARGS)
