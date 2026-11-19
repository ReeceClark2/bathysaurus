#!/usr/bin/env python3
"""Generate every density-inversion dataset in one run.

Datasets = [[cases]] x [receivers.sets] in the TOML (3 x 4 = 12 by default).

The simulation depends only on the density profile and the bathymetry, so each
profile is simulated ONCE per bathymetry (flat, seamount). Pressure is extracted
from that run at every receiver used by any dataset; noise is added afterwards.
Consequences:
  * 2 x 140 simulations instead of 12 x 140.
  * All 12 datasets share the same train/validation/test profiles (same seed as
    before), so differences between datasets come only from receivers, noise
    and bathymetry.
  * A receiver at a given x gets the same noise realization in every dataset
    that contains it.

Simulations run flat bathymetry first, then seamount. Each (bathymetry, split)
is written out to its datasets as soon as all of its runs finish.

Usage:
    python generate_datasets.py generate_datasets.toml
    python generate_datasets.py generate_datasets.toml --resume   # after a crash or walltime limit
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

def bottom_height(x, spec, H):
    """Sea-floor height z(x) [m]; must match bottom_function in forward_model.jl."""
    x = np.asarray(x, dtype=float)
    kind = spec.get("type", "flat")
    if kind == "flat":
        return np.full_like(x, -H)
    if kind == "gaussian":
        return -H + float(spec["height"]) * np.exp(-0.5 * ((x - float(spec["center"])) / float(spec["width"])) ** 2)
    raise ValueError(f"Unknown bathymetry type {kind!r}")


def build_plan(cfg):
    grid = cfg["grid"]
    Nx, Nz = int(grid["Nx"]), int(grid["Nz"])
    Lx, H = float(grid["Lx"]), float(grid["H"])
    if Nx < 1 or Nz < 1 or Lx <= 0 or H <= 0:
        raise ValueError("Check grid.Nx, Nz, Lx, H")
    dz = H / Nz
    x_centers = (np.arange(Nx) + 0.5) * Lx / Nx
    z_centers = -H + (np.arange(Nz) + 0.5) * dz

    receiver_cfg = cfg["receivers"]
    receiver_sets = {name: [float(x) for x in xs] for name, xs in receiver_cfg["sets"].items()}
    if not receiver_sets or any(not xs for xs in receiver_sets.values()):
        raise ValueError("Every receiver set needs at least one receiver")
    if ("depth_fraction" in receiver_cfg) == ("z" in receiver_cfg):
        raise ValueError("Set exactly one of receivers.depth_fraction or receivers.z")
    union_x = sorted({x for xs in receiver_sets.values() for x in xs})

    bathymetry_tables = cfg.get("bathymetry", {})
    cases = cfg["cases"]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Case names must be unique")

    geometries = {}
    datasets = []
    for case in cases:
        bathy = case["bathymetry"]
        if bathy not in bathymetry_tables and bathy != "flat":
            raise ValueError(f"Case {case['name']!r} uses unknown bathymetry {bathy!r}")
        spec = bathymetry_tables.get(bathy, {"type": "flat"})
        if spec.get("type", "flat") == "gaussian":
            if not (0 < float(spec["center"]) < Lx and 0 < float(spec["height"]) < H and float(spec["width"]) > 0):
                raise ValueError(f"Invalid seamount parameters for {bathy!r}")

        level = float(case.get("noise_relative_std", 0.0))
        if not np.isfinite(level) or level < 0:
            raise ValueError(f"noise_relative_std must be >= 0 for case {case['name']!r}")

        if bathy not in geometries:
            bed = bottom_height(union_x, spec, H)
            if "depth_fraction" in receiver_cfg:
                fraction = float(receiver_cfg["depth_fraction"])
                if not 0 < fraction < 1:
                    raise ValueError("receivers.depth_fraction must lie in (0, 1)")
                rz = fraction * bed
            else:
                rz = np.full_like(bed, float(receiver_cfg["z"]))
            for x, z, b in zip(union_x, rz, bed):
                if not x_centers[0] <= x <= x_centers[-1]:
                    raise ValueError(f"Receiver x={x} is outside the x-center range")
                if not b + dz <= z < 0:
                    raise ValueError(
                        f"Receiver (x={x}, z={z:.3f}) is not in the water column for bathymetry "
                        f"{bathy!r} (sea floor at z={b:.3f}; need at least one cell, {dz:.4f} m, above it)")
            geometries[bathy] = {
                "spec": spec,
                "receivers": [(float(x), float(z)) for x, z in zip(union_x, rz)],
                "datasets": [],
            }

        for set_name, xs in receiver_sets.items():
            dataset = {
                "name": f"{len(datasets) + 1:02d}_{case['name']}_{set_name}",
                "case": case["name"],
                "bathymetry": bathy,
                "noise_relative_std": level,
                "columns": [union_x.index(x) for x in xs],
            }
            datasets.append(dataset)
            geometries[bathy]["datasets"].append(dataset)

    return {"z": z_centers, "H": H, "geometries": geometries, "datasets": datasets}


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
    """Full pressure [Pa] at each receiver, relative to atmospheric at the surface.

    full = g * integral(rho dz) from the receiver to the surface   (hydrostatic part)
         + p_nh from the model (rigid-lid surface pressure + nonhydrostatic pressure).
    Returns t, z, full pressure (T, R) and the hydrostatic part alone (T, R).
    """
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        time_name = "time" if "time" in ds else "t"
        t = ds[time_name].to_numpy()
        x = ds["x"].to_numpy()
        z = ds["z"].to_numpy()
        rho = ds["rho"].transpose(time_name, "x", "z").to_numpy()
        if "p_nh" not in ds:
            raise ValueError(f"{nc_path} has no p_nh variable; use the updated forward_model.jl")
        p_nh = ds["p_nh"].transpose(time_name, "x", "z").to_numpy()
        bed = ds["bottom_height"].to_numpy() if "bottom_height" in ds else np.full_like(x, z[0] - 0.5 * (z[1] - z[0]))

    if not np.all(np.isfinite(rho)) or not np.all(np.diff(t) > 0):
        raise ValueError("Simulation contains non-finite density or invalid times")

    hydrostatic, nonhydrostatic = [], []
    for receiver_x, receiver_z in receivers:
        ix = int(np.argmin(np.abs(x - receiver_x)))
        if not bed[ix] < receiver_z <= 0:
            raise ValueError(f"Receiver ({receiver_x}, {receiver_z}) is below the sea floor ({bed[ix]:.3f})")
        z_nodes = np.concatenate(([receiver_z], z[(z > receiver_z) & (z < 0)], [0.0]))
        rho_nodes = np.vstack([np.interp(z_nodes, z, profile) for profile in rho[:, ix, :]])
        hydrostatic.append(g * np.sum(0.5 * (rho_nodes[:, :-1] + rho_nodes[:, 1:]) * np.diff(z_nodes), axis=1))
        nonhydrostatic.append(np.array([np.interp(receiver_z, z, profile) for profile in p_nh[:, ix, :]]))

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
                "bathymetry": {"name": dataset["bathymetry"], **geometry["spec"]},
                "noise_relative_std": dataset["noise_relative_std"],
                "receivers": [{"x": x, "z": z} for x, z in receivers],
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

        # Noise scale: RMS pressure change over all receivers of this sample.
        rms_change = float(np.sqrt(np.mean((pressure - pressure[:1]) ** 2)))
        # Same seed key for every dataset -> receiver at x sees the same realization everywhere.
        unit_noise = np.random.default_rng([int(seed), split_index, run]).standard_normal(pressure.shape)
        target = density[None, None, :].astype(np.float32)

        for dataset in geometry["datasets"]:
            columns = dataset["columns"]
            sigma = dataset["noise_relative_std"] * rms_change
            data = pressure[:, columns] + sigma * unit_noise[:, columns]
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
            writer.writerow(("run", "b", "c", "d", "rms_pressure_change_pa", "noise_std_pa"))
            for run, b, c, d, rms_change, sigma, data_path, model_path in records[dataset["name"]]:
                annotations.write(f"{data_path.resolve()}\t{model_path.resolve()}\n")
                writer.writerow((run, b, c, d, rms_change, sigma))


# ------------------------------------------------------------------------ main

def preflight(pool, plan, cfg, config_path, data_dir, profiles, threads, g):
    """Short run per bathymetry so setup errors surface in minutes, not hours."""
    first = next(((split, rows[0]) for split, rows in profiles.items() if rows), None)
    if first is None:
        return
    split, (run, *_ ) = first
    profile_path = data_dir / "profiles" / split / f"profile_{run:06d}.ascii"

    impulse = cfg["impulse"]
    stop_time = min(float(cfg["run"]["stop_time"]),
                    float(impulse["t0"]) + 2 * float(impulse["tau"]) + 2 * float(cfg["output"]["interval"]))

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
        print(f"Preflight {name}: OK, {len(t)} frames; RMS pressure change so far {rms_change:.3g} Pa, "
              f"RMS nonhydrostatic + lid part {rms_nh:.3g} Pa", flush=True)


def generate(config, resume=False, run_preflight=True):
    config_path = Path(config).resolve()
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

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
        if saved_config.exists() and saved_config.read_bytes() != config_path.read_bytes():
            raise ValueError(f"{config_path} differs from {saved_config}, which this dataset was started with")
    data_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, saved_config)

    write_profiles(profiles, plan["z"], data_dir)
    write_dataset_metadata(plan, data_dir)

    total_runs = len(plan["geometries"]) * sum(len(rows) for rows in profiles.values())
    print(f"{len(plan['datasets'])} datasets from {total_runs} simulations "
          f"({len(plan['geometries'])} bathymetries x {total_runs // len(plan['geometries'])} profiles); "
          f"up to {max_workers} concurrent Julia runs, {threads} threads each", flush=True)
    for geometry_name, geometry in plan["geometries"].items():
        print(f"  {geometry_name}: " + ", ".join(d["name"] for d in geometry["datasets"]), flush=True)

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
    parser.add_argument("config", help="Path to generate_datasets.toml")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run in ./data (finished simulations are kept)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the short test run per bathymetry before production")
    args = parser.parse_args()
    generate(args.config, resume=args.resume, run_preflight=not args.skip_preflight)
    