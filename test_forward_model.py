#!/usr/bin/env python3
"""Run the forward model for ONE density profile and animate the pressure perturbation.

Runs locally (Julia + Oceananigans installed, Project.toml in this folder) and saves an
animated GIF of the pressure perturbation p(x, z, t) - p(x, z, 0) across the tank, with
isopycnals, the receivers, the bathymetry and (with a free surface) the surface height.

Usage, from the bathysaurus folder:
    python animate_profile.py forward_model.toml                       # train profile 1, flat tank
    python animate_profile.py forward_model.toml --bathymetry seamount --run 5
    python animate_profile.py forward_model.toml --b 5 --c -1.0 --d 0.1  # your own profile
    python animate_profile.py forward_model.toml --stop-time 120         # shorter test run
    python animate_profile.py forward_model.toml --reuse                 # re-animate the last run

The run uses --threads Julia threads (default 8). Output goes to ./animation/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import tomllib
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

try:  # the dataset script may be called forward_model.py or generate_datasets.py
    import forward_model as G
except ImportError:
    import generate_datasets as G

ROOT = Path(__file__).resolve().parent


def pressure_field(nc_path, g):
    """Full pressure field [Pa] relative to the surface, and its change since t = 0."""
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        t = ds["t"].to_numpy()
        x = ds["x"].to_numpy()
        z = ds["z"].to_numpy()
        rho = ds["rho"].transpose("t", "x", "z").to_numpy()
        p_nh = ds["p_nh"].transpose("t", "x", "z").to_numpy()
        bed = ds["bottom_height"].to_numpy()
        eta = ds["eta"].transpose("t", "x").to_numpy() if "eta" in ds else None

    # g * integral of rho from each cell centre up to z = 0 (trapezoid between centres,
    # top half-cell with the top value), then add the model's p_nh
    dz = np.diff(z)
    top = rho[:, :, -1] * (0.0 - z[-1])
    layers = 0.5 * (rho[:, :, 1:] + rho[:, :, :-1]) * dz
    hydro = np.concatenate([np.cumsum(layers[:, :, ::-1], axis=2)[:, :, ::-1], np.zeros_like(top)[..., None]], axis=2)
    hydro = g * (hydro + top[..., None])
    pressure = hydro + p_nh

    solid = z[None, :] < bed[:, None]               # cells inside the seamount
    change = pressure - pressure[:1]
    change[:, solid] = np.nan
    rho = rho.copy()
    rho[:, solid] = np.nan
    return t, x, z, rho, change, bed, eta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="your dataset TOML")
    parser.add_argument("--bathymetry", default="flat")
    parser.add_argument("--split", default="train", choices=G.SPLITS)
    parser.add_argument("--run", type=int, default=1, help="profile number within the split")
    parser.add_argument("--b", type=float, help="use your own profile instead (needs --c and --d too)")
    parser.add_argument("--c", type=float)
    parser.add_argument("--d", type=float)
    parser.add_argument("--stop-time", type=float, help="shorter run than the TOML's stop_time")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--every", type=int, default=2, help="animate every N-th output frame")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--reuse", action="store_true", help="skip the simulation, re-animate the saved run")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = tomllib.loads(config_path.read_text())
    plan = G.build_plan(cfg)
    if args.bathymetry not in plan["geometries"]:
        raise ValueError(f"No case uses bathymetry {args.bathymetry!r}; available: {list(plan['geometries'])}")
    geometry = plan["geometries"][args.bathymetry]
    g = float(cfg["physics"]["g"])
    z = plan["z"]

    # The density profile
    if args.b is not None:
        if args.c is None or args.d is None:
            raise ValueError("Give --b, --c and --d together")
        b, c, d = args.b, args.c, args.d
        density = float(cfg["physics"]["rho_ref"]) - b * np.tanh((z - c) / d)
        label = f"custom_b{b:g}_c{c:g}_d{d:g}"
    else:
        rows = G.draw_profiles(cfg, z)[args.split]
        if not 1 <= args.run <= len(rows):
            raise ValueError(f"--run must be between 1 and {len(rows)}")
        _, b, c, d, density = rows[args.run - 1]
        label = f"{args.split}_{args.run:06d}"

    out = ROOT / "animation"
    out.mkdir(exist_ok=True)
    stem = f"{args.bathymetry}_{label}"
    npz_path = out / f"{stem}.npz"
    nc_path = npz_path.with_suffix(".nc")

    if not args.reuse:
        for path in (npz_path, nc_path):
            path.unlink(missing_ok=True)
        profile_path = out / f"{stem}_profile.ascii"
        np.savetxt(profile_path, np.column_stack((z, density)), fmt="%.12f")
        print(f"Running Julia ({args.threads} threads) for b={b:.3f}, c={c:.3f}, d={d:.3f} "
              f"on {args.bathymetry} bathymetry... (log: {npz_path.with_suffix('.log')})", flush=True)
        G.simulate(config_path, args.bathymetry, profile_path, geometry["receivers"], npz_path,
                   args.threads, g, keep_netcdf=True, stop_time=args.stop_time)
    elif not nc_path.exists():
        raise FileNotFoundError(f"{nc_path} not found; run once without --reuse")

    t, x, zc, rho, change, bed, eta = pressure_field(nc_path, g)
    frames = list(range(0, len(t), max(1, args.every)))
    finite = np.abs(change[np.isfinite(change)])
    limit = np.percentile(finite, 99.5) if finite.size else 1.0
    limit = limit if limit > 0 else 1.0
    receivers = np.array(geometry["receivers"])

    # Figure: surface height on top (if any), pressure change below
    rows = 2 if eta is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 6 if eta is not None else 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3] if eta is not None else [1]}, squeeze=False)
    ax = axes[-1, 0]
    mesh = ax.pcolormesh(x, zc, change[0].T, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    fig.colorbar(mesh, ax=list(axes[:, 0]), label="p(t) − p(0) [Pa]", pad=0.01)  # keeps panels aligned
    if np.any(bed > zc[0]):
        ax.fill_between(x, -float(cfg["grid"]["H"]), bed, color="0.45", zorder=3)
    ax.plot(receivers[:, 0], receivers[:, 1], "kv", ms=7, zorder=4, label="receivers")
    levels = np.linspace(np.nanmin(rho[0]), np.nanmax(rho[0]), 9)[1:-1]
    contours = [ax.contour(x, zc, rho[0].T, levels=levels, colors="k", linewidths=0.6, alpha=0.6)]
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="lower left", fontsize=8)
    title = fig.suptitle("")

    if eta is not None:
        ax_eta = axes[0, 0]
        eta_mm = 1e3 * eta
        eta_lim = max(np.abs(eta_mm).max(), 1e-6) * 1.1
        (line,) = ax_eta.plot(x, eta_mm[0], color="tab:blue")
        ax_eta.set_ylim(-eta_lim, eta_lim)
        ax_eta.set_ylabel("η [mm]")
        ax_eta.axhline(0, color="0.6", lw=0.5)

    def update(i):
        k = frames[i]
        mesh.set_array(change[k].T.ravel())
        contours[0].remove()
        contours[0] = ax.contour(x, zc, rho[k].T, levels=levels, colors="k", linewidths=0.6, alpha=0.6)
        if eta is not None:
            line.set_ydata(eta_mm[k])
        title.set_text(f"{args.bathymetry}, b = {b:.2f}, c = {c:.2f} m, d = {d:.2f} m    t = {t[k]:6.1f} s")
        return [mesh, title]

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    gif_path = out / f"{stem}.gif"
    print(f"Writing {len(frames)} frames to {gif_path} ...", flush=True)
    animation.save(gif_path, writer=PillowWriter(fps=args.fps))
    print(f"Saved {gif_path}\nColour scale: ±{limit:.3g} Pa (99.5th percentile of |p − p0|)")

    # Also a still of the receiver traces, for a quick look at the signal level
    with np.load(npz_path) as f:
        p = f["pressure"]
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for i, (rx, rz) in enumerate(geometry["receivers"]):
        ax2.plot(t, p[:, i] - p[0, i], lw=1, label=f"x = {rx:g} m")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("p(t) − p(0) [Pa]")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / f"{stem}_receivers.png", dpi=150)
    print(f"Saved {out / f'{stem}_receivers.png'}  (peak |p − p0| = {np.abs(p - p[:1]).max():.3g} Pa)")


if __name__ == "__main__":
    main()
    