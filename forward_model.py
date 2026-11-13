from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import shutil
import subprocess
import tomllib

import numpy as np
import xarray as xr

import os
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from tempfile import TemporaryDirectory
import csv
import shutil


def forward_model(receiver_positions, config, density, julia_output, data_output, run_index, cores_per_forward_model):
    root = Path(__file__).resolve().parent
    config_path = Path(config).resolve()
    density_path = Path(density).resolve()
    julia_output_path = Path(julia_output).resolve()
    save_dir = Path(data_output).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    subprocess.run(
        ["julia", f"--threads={cores_per_forward_model}", f"--project={root}", str(root / "forward_model.jl"), str(config_path), str(density_path), str(julia_output_path)],
        cwd=root,
        env=env,
        check=True,
    )

    with config_path.open("rb") as f:
        g = float(tomllib.load(f)["physics"]["g"])

    with xr.open_dataset(julia_output_path) as ds:
        time_name = "time" if "time" in ds else "t"
        t = ds[time_name].to_numpy()
        x = ds["x"].to_numpy()
        z = ds["z"].to_numpy()
        rho = ds["rho"].transpose(time_name, "x", "z").to_numpy()

    if not np.all(np.isfinite(rho)) or not np.all(np.diff(t) > 0):
        raise ValueError("Simulation contains non-finite density or invalid times")
    if not receiver_positions:
        raise ValueError("At least one receiver is required")

    pressure_columns = []

    for receiver_x, receiver_z in receiver_positions:
        if receiver_x < x[0] or receiver_x > x[-1]:
            raise ValueError(f"Receiver x={receiver_x} is outside the x-center range")
        if receiver_z < z[0] or receiver_z > 0:
            raise ValueError(f"Receiver z={receiver_z} is outside the water column")

        ix = int(np.argmin(np.abs(x - receiver_x)))
        z_nodes = np.concatenate(([receiver_z], z[(z > receiver_z) & (z < 0)], [0.0]))
        rho_nodes = np.vstack([np.interp(z_nodes, z, profile) for profile in rho[:, ix, :]])
        p = g * np.sum(0.5 * (rho_nodes[:, :-1] + rho_nodes[:, 1:]) * np.diff(z_nodes), axis=1)
        pressure_columns.append(p)

    pressure = np.stack(pressure_columns, axis=-1)
    input_data = pressure[None, None, :, :].astype(np.float32)

    profile = np.loadtxt(density_path, comments="#")
    if profile.shape != (len(z), 2):
        raise ValueError(f"Expected profile shape {(len(z), 2)}; got {profile.shape}")
    if not np.allclose(profile[:, 0], z, rtol=0, atol=1e-6):
        raise ValueError("Profile depths do not match the output grid")
    if not np.all(np.isfinite(input_data)):
        raise ValueError("Computed pressure contains non-finite values")

    target = profile[:, 1][None, None, :].astype(np.float32)
    data_path = save_dir / f"data_{run_index:06d}.npy"
    model_path = save_dir / f"model_{run_index:06d}.npy"

    np.save(data_path, input_data)
    np.save(model_path, target)

    return data_path, model_path, t, z


def run_sample(receiver_positions, config_path, profile_path, split_dir, run_index, cores_per_forward_model):
    with TemporaryDirectory(prefix=f"forward_{run_index:06d}_", dir=split_dir) as temporary_dir:
        julia_output = Path(temporary_dir) / "density.nc"
        return forward_model(receiver_positions, config_path, profile_path, julia_output, split_dir, run_index, cores_per_forward_model)


def forward_model_orchestrator(config):
    root = Path(__file__).resolve().parent
    config_path = Path(config).resolve()

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    data_dir = root / "data"
    if data_dir.exists():
        raise FileExistsError(f"{data_dir} already exists; refusing to overwrite a dataset")
    if not (root / "forward_model.jl").is_file():
        raise FileNotFoundError(root / "forward_model.jl")

    Nz = int(cfg["grid"]["Nz"])
    H = float(cfg["grid"]["H"])
    rho_ref = float(cfg["physics"]["rho_ref"])
    seed = int(cfg["dataset"]["seed"])
    receivers = [(float(r["x"]), float(r["z"])) for r in cfg["receivers"]]
    counts = {
        "train": int(cfg["dataset"]["train"]),
        "validation": int(cfg["dataset"]["validation"]),
        "test": int(cfg["dataset"]["test"]),
    }

    allocated_cores = int(cfg["parallel"]["allocated_cores"])
    cores_per_forward_model = int(cfg["parallel"]["cores_per_forward_model"])

    if Nz < 1 or H <= 0 or rho_ref <= 0 or not receivers:
        raise ValueError("Check Nz, H, rho_ref, and receivers in the TOML")
    if any(count < 0 for count in counts.values()):
        raise ValueError("Dataset split counts cannot be negative")
    if allocated_cores < 1 or cores_per_forward_model < 1:
        raise ValueError("Parallel core counts must be positive")
    if cores_per_forward_model > allocated_cores:
        raise ValueError("cores_per_forward_model exceeds allocated_cores")

    max_workers = allocated_cores // cores_per_forward_model

    ranges = {}
    for name in ("b", "c", "d"):
        low = float(cfg["profile"][name]["min"])
        high = float(cfg["profile"][name]["max"])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid range for {name}")
        ranges[name] = (low, high)

    if ranges["b"][0] <= 0 or ranges["d"][0] <= 0:
        raise ValueError("b and d must be strictly positive")
    if not (-H < ranges["c"][0] < ranges["c"][1] < 0):
        raise ValueError("The c range must lie strictly inside (-H, 0)")

    rng = np.random.default_rng(seed)
    z = -H + (np.arange(Nz) + 0.5) * H / Nz

    data_dir.mkdir()
    shutil.copy2(config_path, data_dir / "config.toml")
    print(f"Using up to {max_workers} concurrent Julia runs, {cores_per_forward_model} Julia threads each", flush=True)

    for split, count in counts.items():
        split_dir = data_dir / split
        profiles_dir = split_dir / "profiles"
        profiles_dir.mkdir(parents=True)

        reference_time = None
        reference_z = None
        next_to_submit = 1
        next_to_record = 1
        pending = {}
        completed = {}

        with (data_dir / f"{split}.txt").open("w", encoding="utf-8") as annotations:
            with (split_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as parameter_file:
                writer = csv.writer(parameter_file)
                writer.writerow(("run", "b", "c", "d"))

                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    while next_to_submit <= count or pending:
                        while next_to_submit <= count and len(pending) < max_workers:
                            run_index = next_to_submit
                            b = rng.uniform(*ranges["b"])
                            c = rng.uniform(*ranges["c"])
                            d = rng.uniform(*ranges["d"])

                            density = rho_ref - b * np.tanh((z - c) / d)
                            profile_path = profiles_dir / f"profile_{run_index:06d}.ascii"
                            np.savetxt(profile_path, np.column_stack((z, density)), fmt="%.12f")

                            future = pool.submit(
                                run_sample, receivers, config_path, profile_path,
                                split_dir, run_index, cores_per_forward_model
                            )
                            pending[future] = (run_index, b, c, d)
                            next_to_submit += 1

                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)

                        for future in finished:
                            run_index, b, c, d = pending.pop(future)

                            try:
                                data_path, model_path, t, output_z = future.result()
                            except Exception as exc:
                                raise RuntimeError(f"{split} run {run_index} failed") from exc

                            if reference_time is None:
                                reference_time = t.copy()
                                reference_z = output_z.copy()
                                np.save(split_dir / "time.npy", reference_time)
                                np.save(split_dir / "z.npy", reference_z)
                            elif not np.array_equal(t, reference_time) or not np.array_equal(output_z, reference_z):
                                raise ValueError(f"Grid or time samples changed during {split} run {run_index}")

                            completed[run_index] = (data_path, model_path, b, c, d)

                        while next_to_record in completed:
                            data_path, model_path, b, c, d = completed.pop(next_to_record)
                            annotations.write(f"{data_path.resolve()}\t{model_path.resolve()}\n")
                            annotations.flush()
                            writer.writerow((next_to_record, b, c, d))
                            parameter_file.flush()
                            print(f"{split}: completed {next_to_record}/{count}", flush=True)
                            next_to_record += 1

    print(f"Dataset saved to {data_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the combined TOML configuration")
    args = parser.parse_args()

    forward_model_orchestrator(args.config)
