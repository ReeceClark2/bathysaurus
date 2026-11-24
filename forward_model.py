#!/usr/bin/env python3
"""Generate every density-inversion dataset for the towed-seamount tank in one run.

Datasets = [[cases]] x [receivers.sets] in the TOML (2 noise levels x 4 receiver sets = 8).

Each density profile is simulated ONCE (forward_model.jl: closed 2D tank, rigid lid, a ridge
towed along the floor). Receivers ride with the seamount at fixed offsets from its centre;
pressure is read at their moving positions from every saved frame. Noise is added afterwards,
so all datasets share the same simulations and train/validation/test profiles, and a receiver
at a given offset gets the same noise realization in every dataset that contains it.

Usage:
    python forward_model.py forward_model.toml
    python forward_model.py forward_model.toml --resume   # after a crash or walltime limit
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tomllib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- plan

GEOMETRY = "seamount"   # single geometry; its simulations live in data/simulations/seamount/


def tow_position(t, seamount):
    """Seamount centre x(t) [m]; must match tow_position in forward_model.jl.

    Speed = speed * sin^2(pi (t - t_start) / duration) during the motion window, 0 outside it.
    """
    t = np.asarray(t, dtype=float)
    start_x, speed = float(seamount["start_x"]), float(seamount["speed"])
    t_start, duration = float(seamount["t_start"]), float(seamount["duration"])
    s = np.clip(t - t_start, 0.0, duration)
    return start_x - speed * (s / 2 - duration / (4 * np.pi) * np.sin(2 * np.pi * s / duration))


def build_plan(cfg):
    grid = cfg["grid"]
    Nx, Nz = int(grid["Nx"]), int(grid["Nz"])
    Lx, H = float(grid["Lx"]), float(grid["H"])
    if Nx < 1 or Nz < 1 or Lx <= 0 or H <= 0:
        raise ValueError("Check grid.Nx, Nz, Lx, H")
    dx, dz = Lx / Nx, H / Nz
    z_centers = -H + (np.arange(Nz) + 0.5) * dz

    seamount = cfg["seamount"]
    height, width = float(seamount["height"]), float(seamount["width"])
    if not (0 < height < H and width > 0):
        raise ValueError("Check seamount.height (inside (0, H)) and seamount.width (> 0)")
    stop_time = float(cfg["run"]["stop_time"])
    t_start, duration = float(seamount["t_start"]), float(seamount["duration"])
    if t_start < 0 or duration <= 0:
        raise ValueError("Check seamount.t_start (>= 0) and seamount.duration (> 0)")
    if t_start + duration > stop_time:
        raise ValueError(f"The seamount is still moving at stop_time ({t_start + duration:g} s > {stop_time:g} s)")
    path = tow_position([0.0, stop_time], seamount)
    if path.min() - width / 2 < dx or path.max() + width / 2 > Lx - dx:
        raise ValueError(f"The seamount would touch a wall (centre from {path[0]:.2f} to {path[1]:.2f} m)")

    receiver_cfg = cfg["receivers"]
    receiver_sets = {name: [float(x) for x in xs] for name, xs in receiver_cfg["sets"].items()}
    if not receiver_sets or any(not xs for xs in receiver_sets.values()):
        raise ValueError("Every receiver set needs at least one receiver")
    height_above_bottom = float(receiver_cfg["height_above_bottom"])
    if not dz <= height_above_bottom < H:
        raise ValueError(f"receivers.height_above_bottom must be at least one cell ({dz:.4f} m)")
    offsets = sorted({x for xs in receiver_sets.values() for x in xs})
    clearance = width / 2 + 0.1
    for offset in offsets:
        if abs(offset) < clearance:
            raise ValueError(f"Receiver offset {offset} m is on or next to the seamount; keep |offset| >= {clearance:.2f} m "
                             "(the pressure next to the moving solid is less accurate)")
        positions = path + offset
        if positions.min() < dx or positions.max() > Lx - dx:
            raise ValueError(f"Receiver offset {offset} m leaves the tank during the run "
                             f"(x from {positions[0]:.2f} to {positions[1]:.2f} m)")
    receivers = [(offset, -H + height_above_bottom) for offset in offsets]

    cases = cfg["cases"]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Case names must be unique")

    geometry = {"spec": {"type": "towed", **seamount}, "receivers": receivers, "datasets": []}
    datasets = []
    for case in cases:
        if "noise_relative_std" in case:
            raise ValueError(f"Case {case['name']!r}: use noise_std_pa (fixed noise standard deviation in Pa)")
        noise_std_pa = float(case.get("noise_std_pa", 0.0))
        if not np.isfinite(noise_std_pa) or noise_std_pa < 0:
            raise ValueError(f"noise_std_pa must be >= 0 for case {case['name']!r}")
        for set_name, xs in receiver_sets.items():
            dataset = {
                "name": f"{len(datasets) + 1:02d}_{case['name']}_{set_name}",
                "case": case["name"],
                "bathymetry": GEOMETRY,
                "noise_std_pa": noise_std_pa,
                "columns": [offsets.index(x) for x in xs],
            }
            datasets.append(dataset)
            geometry["datasets"].append(dataset)

    return {"z": z_centers, "H": H, "geometries": {GEOMETRY: geometry}, "datasets": datasets}


def draw_profiles(cfg, z):
    """Same draws, in the same order, as the original forward_model.py."""
    rho_ref = float(cfg["physics"]["rho_ref"])
    H = float(cfg["grid"]["H"])
    if rho_ref <= 0:
        raise ValueError("rho_ref must be positive")

    ranges = {}
    for name in ("b", "c", "d"):
        low, high = float(cfg["profile"][name]["min"]), float(cfg["profile"][name]["max"])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid range for {name}")
        ranges[name] = (low, high)
    if ranges["b"][0] <= 0 or ranges["d"][0] <= 0:
        raise ValueError("b and d must be strictly positive")
    if not -H < ranges["c"][0] < ranges["c"][1] < 0:
        raise ValueError("The c range must lie strictly inside (-H, 0)")

    rng = np.random.default_rng(int(cfg["dataset"]["seed"]))
    profiles = {}
    for split in SPLITS:
        count = int(cfg["dataset"][split])
        if count < 0:
            raise ValueError("Dataset split counts cannot be negative")
        rows = []
        for run in range(1, count + 1):
            b = rng.uniform(*ranges["b"])
            c = rng.uniform(*ranges["c"])
            d = rng.uniform(*ranges["d"])
            rows.append((run, b, c, d, rho_ref - b * np.tanh((z - c) / d)))
        profiles[split] = rows
    return profiles


# ------------------------------------------------------------------ simulation

def receiver_pressure(nc_path, receivers, g):
    """Full pressure [Pa] at receivers riding with the seamount, relative to the surface.

    receivers: (offset from the seamount centre [m], z [m]) pairs. At each frame the receiver
    sits at x = seamount_x(t) + offset; fields are interpolated linearly between grid columns.
    full = g * integral(rho dz) from the receiver to the surface + p_nh from the model.
    Returns t, z, full pressure (T, R) and the hydrostatic part alone (T, R).
    """
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        t = ds["t"].to_numpy()
        x = ds["x"].to_numpy()
        z = ds["z"].to_numpy()
        seamount_x = ds["seamount_x"].to_numpy()
        rho = ds["rho"].transpose("t", "x", "z").to_numpy()
        p_nh = ds["p_nh"].transpose("t", "x", "z").to_numpy()

    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(p_nh)) or not np.all(np.diff(t) > 0):
        raise ValueError("Simulation contains non-finite fields or invalid times")

    frames = np.arange(len(t))
    hydrostatic, nonhydrostatic = [], []
    for offset, receiver_z in receivers:
        xr_t = seamount_x + offset                                   # receiver position per frame
        i = np.clip(np.searchsorted(x, xr_t) - 1, 0, len(x) - 2)
        w = np.clip((xr_t - x[i]) / (x[i + 1] - x[i]), 0.0, 1.0)[:, None]
        rho_col = (1 - w) * rho[frames, i, :] + w * rho[frames, i + 1, :]       # (T, Nz)
        pnh_col = (1 - w) * p_nh[frames, i, :] + w * p_nh[frames, i + 1, :]

        z_nodes = np.concatenate(([receiver_z], z[(z > receiver_z) & (z < 0)], [0.0]))
        rho_nodes = np.vstack([np.interp(z_nodes, z, column) for column in rho_col])
        hydrostatic.append(g * np.sum(0.5 * (rho_nodes[:, :-1] + rho_nodes[:, 1:]) * np.diff(z_nodes), axis=1))
        nonhydrostatic.append(np.array([np.interp(receiver_z, z, column) for column in pnh_col]))

    hydrostatic = np.stack(hydrostatic, axis=-1)
    pressure = hydrostatic + np.stack(nonhydrostatic, axis=-1)
    if not np.all(np.isfinite(pressure)):
        raise ValueError("Computed pressure contains non-finite values")
    return t, z, pressure, hydrostatic


def simulate(config_path, bathymetry, profile_path, receivers, out_path, threads, g,
             keep_netcdf=False, stop_time=None):
    """Run one Julia simulation and save clean receiver pressures to out_path (.npz)."""
    out_path = Path(out_path)
    log_path = out_path.with_suffix(".log")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    with TemporaryDirectory(prefix=f"{out_path.stem}_", dir=out_path.parent) as tmp:
        nc_path = Path(tmp) / "density.nc"
        cmd = ["julia", f"--threads={threads}", f"--project={ROOT}", str(ROOT / "forward_model.jl"),
               str(config_path), str(profile_path), str(nc_path), bathymetry]
        if stop_time is not None:
            cmd.append(repr(float(stop_time)))
        with log_path.open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Julia exited with code {result.returncode}; see {log_path}")

        t, z, pressure, hydrostatic = receiver_pressure(nc_path, receivers, g)
        if keep_netcdf:
            shutil.move(nc_path, out_path.with_suffix(".nc"))

    # pressure = full pressure (used for the datasets); hydrostatic kept only for diagnostics.
    partial = out_path.with_name(out_path.stem + ".partial.npz")
    np.savez(partial, t=t, z=z, pressure=pressure, pressure_hydrostatic=hydrostatic,
             receivers=np.asarray(receivers))
    os.replace(partial, out_path)  # the .npz appears only once complete; used for --resume
    return out_path


# --------------------------------------------------------------------- outputs

def write_profiles(profiles, z, data_dir):
    for split, rows in profiles.items():
        split_dir = data_dir / "profiles" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("run", "b", "c", "d"))
            for run, b, c, d, density in rows:
                writer.writerow((run, b, c, d))
                np.savetxt(split_dir / f"profile_{run:06d}.ascii", np.column_stack((z, density)), fmt="%.12f")


def write_dataset_metadata(plan, data_dir):
    for geometry in plan["geometries"].values():
        for dataset in geometry["datasets"]:
            dataset_dir = data_dir / "datasets" / dataset["name"]
            for split in SPLITS:
                (dataset_dir / split).mkdir(parents=True, exist_ok=True)
            receivers = [geometry["receivers"][i] for i in dataset["columns"]]
            metadata = {
                "name": dataset["name"],
                "case": dataset["case"],
                "seamount": geometry["spec"],
                "noise_std_pa": dataset["noise_std_pa"],
                "receivers": [{"offset_from_seamount_centre": x, "z": z} for x, z in receivers],
                "receiver_note": "receivers move with the seamount: x(t) = seamount centre + offset "
                                 "(negative = ahead of it, positive = behind)",
                "data_shape": "(1, 1, n_times, n_receivers), receiver order as listed; full pressure "
                              "(hydrostatic + nonhydrostatic) in Pa, zero at the surface",
                "model_shape": "(1, 1, Nz), density in kg/m^3 at z.npy",
            }
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def finalize_split(plan, geometry_name, split, profiles, data_dir, seed):
    """Turn the clean pressures of one (bathymetry, split) into every dataset that uses them."""
    geometry = plan["geometries"][geometry_name]
    sim_dir = data_dir / "simulations" / geometry_name / split
    split_index = SPLITS.index(split)
    reference_t = None
    records = {d["name"]: [] for d in geometry["datasets"]}

    for run, b, c, d, density in profiles[split]:
        with np.load(sim_dir / f"pressure_{run:06d}.npz") as f:
            t, z, pressure = f["t"], f["z"], f["pressure"]

        if reference_t is None:
            reference_t = t
            if not np.allclose(z, plan["z"], rtol=0, atol=1e-6):
                raise ValueError("Simulation z grid does not match the profile grid")
        elif not np.array_equal(t, reference_t):
            raise ValueError(f"Time samples changed in {geometry_name} {split} run {run}")

        change = pressure - pressure[:1]
        # Same seed key for every dataset -> receiver at x sees the same realization everywhere.
        unit_noise = np.random.default_rng([int(seed), split_index, run]).standard_normal(pressure.shape)
        target = density[None, None, :].astype(np.float32)

        for dataset in geometry["datasets"]:
            columns = dataset["columns"]
            sigma = dataset["noise_std_pa"]  # fixed sensor noise, the same for every sample
            data = pressure[:, columns] + sigma * unit_noise[:, columns]
            # RMS signal (p(t) - p(0)) over this dataset's receivers, for reference
            rms_change = float(np.sqrt(np.mean(change[:, columns] ** 2)))
            out_dir = data_dir / "datasets" / dataset["name"] / split
            data_path = out_dir / f"data_{run:06d}.npy"
            model_path = out_dir / f"model_{run:06d}.npy"
            np.save(data_path, data[None, None, :, :].astype(np.float32))
            np.save(model_path, target)
            records[dataset["name"]].append((run, b, c, d, rms_change, sigma, data_path, model_path))

    for dataset in geometry["datasets"]:
        dataset_dir = data_dir / "datasets" / dataset["name"]
        if reference_t is not None:
            np.save(dataset_dir / split / "time.npy", reference_t)
            np.save(dataset_dir / split / "z.npy", plan["z"])
        with (dataset_dir / f"{split}.txt").open("w", encoding="utf-8") as annotations, \
             (dataset_dir / split / "parameters.csv").open("w", newline="", encoding="utf-8") as params:
            writer = csv.writer(params)
            writer.writerow(("run", "b", "c", "d", "rms_pressure_change_pa", "noise_std_pa", "snr"))
            for run, b, c, d, rms_change, sigma, data_path, model_path in records[dataset["name"]]:
                annotations.write(f"{data_path.resolve()}\t{model_path.resolve()}\n")
                snr = rms_change / sigma if sigma > 0 else float("inf")
                writer.writerow((run, b, c, d, rms_change, sigma, snr))


# ------------------------------------------------------------------------ main

def preflight(pool, plan, cfg, config_path, data_dir, profiles, threads, g):
    """Short run per bathymetry so setup errors surface in minutes, not hours."""
    first = next(((split, rows[0]) for split, rows in profiles.items() if rows), None)
    if first is None:
        return
    split, (run, *_ ) = first
    profile_path = data_dir / "profiles" / split / f"profile_{run:06d}.ascii"

    seamount = cfg["seamount"]  # short run into the start of the motion
    stop_time = min(float(cfg["run"]["stop_time"]),
                    float(seamount["t_start"]) + 0.1 * float(seamount["duration"]) + 2 * float(cfg["output"]["interval"]))

    preflight_dir = data_dir / "preflight"
    shutil.rmtree(preflight_dir, ignore_errors=True)
    preflight_dir.mkdir()

    futures = {
        pool.submit(simulate, config_path, name, profile_path, geometry["receivers"],
                    preflight_dir / f"{name}.npz", threads, g, False, stop_time): name
        for name, geometry in plan["geometries"].items()
    }
    print(f"Preflight: {len(futures)} short run(s) to t = {stop_time:g} s", flush=True)
    for future, name in futures.items():
        try:
            path = future.result()
        except Exception as exc:
            raise RuntimeError(f"Preflight failed for bathymetry {name!r}; no production runs were started") from exc
        with np.load(path) as f:
            t, pressure, hydrostatic = f["t"], f["pressure"], f["pressure_hydrostatic"]
        rms_change = np.sqrt(np.mean((pressure - pressure[:1]) ** 2))
        rms_nh = np.sqrt(np.mean((pressure - hydrostatic) ** 2))
        # 1e4 Pa is about 1 m of water: no realistic tank signal comes near it.
        if not np.isfinite(rms_change) or rms_change > 1e4 or rms_nh > 1e4:
            raise RuntimeError(f"Preflight {name}: implausible pressure (RMS change {rms_change:.3g} Pa, "
                               f"nonhydrostatic + surface part {rms_nh:.3g} Pa); the pressure solve has "
                               f"probably failed. No production runs were started.")
        print(f"Preflight {name}: OK, {len(t)} frames; RMS pressure change so far {rms_change:.3g} Pa, "
              f"RMS nonhydrostatic + lid part {rms_nh:.3g} Pa", flush=True)
        if rms_change == 0:
            print(f"Preflight {name}: warning, no pressure change at all; check the [seamount] motion", flush=True)


def generate(config, resume=False, run_preflight=True, cores=None):
    config_path = Path(config).resolve()
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    if cores is not None:  # e.g. from SLURM_CPUS_PER_TASK; may differ between submissions
        cfg["parallel"]["allocated_cores"] = int(cores)

    if not (ROOT / "forward_model.jl").is_file():
        raise FileNotFoundError(ROOT / "forward_model.jl")

    plan = build_plan(cfg)
    profiles = draw_profiles(cfg, plan["z"])
    g = float(cfg["physics"]["g"])
    seed = int(cfg["dataset"]["seed"])

    parallel = cfg["parallel"]
    allocated_cores = int(parallel["allocated_cores"])
    threads = int(parallel["cores_per_forward_model"])
    keep_netcdf = bool(parallel.get("keep_netcdf", False))
    if allocated_cores < 1 or threads < 1 or threads > allocated_cores:
        raise ValueError("Check parallel.allocated_cores and parallel.cores_per_forward_model")
    max_workers = allocated_cores // threads

    data_dir = ROOT / "data"
    saved_config = data_dir / "config.toml"
    if data_dir.exists():
        if not resume:
            raise FileExistsError(f"{data_dir} already exists; use --resume to continue it, or move it away")
        if saved_config.exists():
            # Everything that affects the simulations must match the original run. [parallel]
            # and [[cases]] (names, noise) may change: datasets are rebuilt from the saved
            # simulations. A case with a new bathymetry simply gets its simulations run.
            with saved_config.open("rb") as f:
                saved = tomllib.load(f)
            changed = sorted(k for k in set(saved) | set(cfg)
                             if k not in ("parallel", "cases") and saved.get(k) != cfg.get(k))
            if changed:
                raise ValueError(f"{config_path} differs from {saved_config}, which this dataset was "
                                 f"started with, in: {', '.join(changed)} (only [parallel] and [[cases]] may change)")
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, saved_config)

    write_profiles(profiles, plan["z"], data_dir)
    write_dataset_metadata(plan, data_dir)
    stale = sorted(p.name for p in (data_dir / "datasets").iterdir()
                   if p.is_dir() and p.name not in {d["name"] for d in plan["datasets"]})
    if stale:
        print(f"Note: data/datasets has folders not in the current TOML (left untouched): "
              f"{', '.join(stale)}", flush=True)

    total_runs = len(plan["geometries"]) * sum(len(rows) for rows in profiles.values())
    print(f"{len(plan['datasets'])} datasets from {total_runs} simulations "
          f"({len(plan['geometries'])} bathymetries x {total_runs // len(plan['geometries'])} profiles); "
          f"up to {max_workers} concurrent Julia runs, {threads} threads each", flush=True)
    for geometry_name, geometry in plan["geometries"].items():
        print(f"  {geometry_name}: " + ", ".join(d["name"] for d in geometry["datasets"]), flush=True)
        print(f"  receiver offsets from the seamount centre [m]: "
              + ", ".join(f"{x:+g}" for x, _ in geometry["receivers"]), flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        if run_preflight:
            preflight(pool, plan, cfg, config_path, data_dir, profiles, threads, g)

        pending = {}
        remaining = {}
        for geometry_name, geometry in plan["geometries"].items():
            for split in SPLITS:
                sim_dir = data_dir / "simulations" / geometry_name / split
                sim_dir.mkdir(parents=True, exist_ok=True)
                for leftover in sim_dir.iterdir():  # from an interrupted earlier attempt
                    if leftover.is_dir():
                        shutil.rmtree(leftover)
                    elif leftover.name.endswith(".partial.npz"):
                        leftover.unlink()

                todo = [run for run, *_ in profiles[split] if not (sim_dir / f"pressure_{run:06d}.npz").exists()]
                remaining[(geometry_name, split)] = len(todo)
                for run in todo:
                    future = pool.submit(
                        simulate, config_path, geometry_name,
                        data_dir / "profiles" / split / f"profile_{run:06d}.ascii",
                        geometry["receivers"], sim_dir / f"pressure_{run:06d}.npz",
                        threads, g, keep_netcdf)
                    pending[future] = (geometry_name, split, run)

        def finish(key):
            finalize_split(plan, *key, profiles, data_dir, seed)
            names = ", ".join(d["name"] for d in plan["geometries"][key[0]]["datasets"])
            print(f"{key[0]} {key[1]}: written to {names}", flush=True)

        for key, count in remaining.items():
            if count == 0:
                finish(key)

        done, to_run = 0, len(pending)
        if resume:
            print(f"Resuming: {total_runs - to_run} simulations already done, {to_run} to run", flush=True)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                geometry_name, split, run = pending.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"{geometry_name} {split} run {run} failed "
                                       f"(rerun with --resume to continue)") from exc
                done += 1
                remaining[(geometry_name, split)] -= 1
                print(f"[{done}/{to_run}] {geometry_name} {split} run {run} done", flush=True)
                if remaining[(geometry_name, split)] == 0:
                    finish((geometry_name, split))

    print(f"All {len(plan['datasets'])} datasets saved to {data_dir / 'datasets'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to forward_model.toml")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in ./data (finished simulations are kept)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the short test run per bathymetry before production")
    parser.add_argument("--cores", type=int,
                        help="Override parallel.allocated_cores (e.g. $SLURM_CPUS_PER_TASK)")
    args = parser.parse_args()
    generate(args.config, resume=args.resume, run_preflight=not args.skip_preflight, cores=args.cores)
