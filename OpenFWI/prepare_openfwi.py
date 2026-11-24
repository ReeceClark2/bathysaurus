#!/usr/bin/env python3
"""Add Bathysaurus datasets to OpenFWI's dataset_config.json.

Run from the OpenFWI folder:
    python prepare_openfwi.py /path/to/bathysaurus/data/datasets

For each dataset folder (01_flat_noiseless_r1, ...) and each pressure mode it adds
an entry named "<dataset>-<mode>", e.g. "07_flat_noisy_r4-dynamic", holding:
  * the normalization range of the pre-processed pressure (from the TRAINING split only),
  * the density range for the labels (rho_ref +- the largest b in the TOML),
  * the shapes the model needs (time samples, receivers, profile length).
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from bathysaurus_openfwi import PressureTransform

MODES = ("dynamic", "anomaly")


def dataset_entry(dataset_dir, cfg, mode):
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    t = np.load(dataset_dir / "train" / "time.npy")
    z = np.load(dataset_dir / "train" / "z.npy")
    g = float(cfg["physics"]["g"])
    rho_ref = float(cfg["physics"]["rho_ref"])
    b_max = float(cfg["profile"]["b"]["max"])

    # "Before the impulse": earlier than t0 - 3 tau (t < 6 s with the default TOML).
    quiet_until = float(cfg["impulse"]["t0"]) - 3 * float(cfg["impulse"]["tau"])
    baseline_samples = max(1, int(np.sum(t < quiet_until)))
    reference_pa = [rho_ref * g * -r["z"] for r in metadata["receivers"]]
    transform = PressureTransform(mode, reference_pa, baseline_samples)

    lines = (dataset_dir / "train.txt").read_text().splitlines()
    if not lines:
        raise ValueError(f"{dataset_dir}/train.txt is empty")
    low, high = np.inf, -np.inf
    for line in lines:
        x = transform(np.load(line.split("\t")[0])[0])
        low, high = min(low, float(x.min())), max(high, float(x.max()))

    return {
        "pressure_mode": mode,
        "data_min": low,
        "data_max": high,
        "label_min": rho_ref - b_max,
        "label_max": rho_ref + b_max,
        "file_size": 1,
        "nt": len(t),
        "ng": len(metadata["receivers"]),
        "n_grid": len(z),
        "baseline_samples": baseline_samples,
        "reference_pa": reference_pa,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets_dir", help="bathysaurus data/datasets folder")
    parser.add_argument("--config", default="dataset_config.json", help="OpenFWI dataset_config.json to update")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    with (datasets_dir.parent / "config.toml").open("rb") as f:
        cfg = tomllib.load(f)

    config_path = Path(args.config)
    entries = json.loads(config_path.read_text()) if config_path.exists() else {}

    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if (p / "metadata.json").is_file()):
        for mode in args.modes:
            name = f"{dataset_dir.name}-{mode}"
            entries[name] = dataset_entry(dataset_dir, cfg, mode)
            e = entries[name]
            print(f"{name}: {e['ng']} receivers, {e['nt']} times, "
                  f"pressure range [{e['data_min']:.3g}, {e['data_max']:.3g}] Pa")

    config_path.write_text(json.dumps(entries, indent=4) + "\n")
    print(f"Updated {config_path}")


if __name__ == "__main__":
    main()
