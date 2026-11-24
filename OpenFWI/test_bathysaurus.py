#!/usr/bin/env python3
"""Evaluate a trained InversionNet1D on a Bathysaurus test split.

Run from the OpenFWI folder, e.g.
    python test_bathysaurus.py -ds 07_flat_noisy_r4-dynamic \
        -ap /path/to/data/datasets/07_flat_noisy_r4 -t test.txt \
        -c Invnet_models/07_flat_noisy_r4/dynamic/checkpoint.pth

Writes next to the checkpoint:
    <split>_predictions.npz   true and predicted profiles [kg/m^3], z, run numbers
    <split>_metrics.csv       per-sample MAE and RMSE [kg/m^3]
and prints the mean errors.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import network
import transforms as T
import bathysaurus_openfwi
from dataset import FWIDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-ds", "--dataset", required=True, help="entry in dataset_config.json")
    parser.add_argument("-ap", "--anno-path", required=True, help="dataset folder")
    parser.add_argument("-t", "--test-anno", default="test.txt")
    parser.add_argument("-c", "--checkpoint", required=True, help="checkpoint.pth written by train.py")
    parser.add_argument("-m", "--model", default="InversionNet1D")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open("dataset_config.json") as f:
        ctx = json.load(f)[args.dataset]
    transform_data, transform_label = bathysaurus_openfwi.build_transforms(ctx)

    anno = Path(args.anno_path) / args.test_anno
    dataset = FWIDataset(str(anno), preload=True, file_size=1,
                         transform_data=transform_data, transform_label=transform_label)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = network.model_dict[args.model](**bathysaurus_openfwi.model_kwargs(ctx))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(network.replace_legacy(checkpoint["model"]))
    model.to(args.device).eval()

    predictions, truths = [], []
    with torch.no_grad():
        for data, label in loader:
            predictions.append(model(data.to(args.device)).cpu().numpy())
            truths.append(label.numpy())
    # Back to kg/m^3
    pred = T.minmax_denormalize(np.concatenate(predictions), ctx["label_min"], ctx["label_max"])[:, 0]
    true = T.minmax_denormalize(np.concatenate(truths), ctx["label_min"], ctx["label_max"])[:, 0]

    def run_number(line, index):
        tail = Path(line.split("\t")[0]).stem.split("_")[-1]  # data_000042 -> 42
        return int(tail) if tail.isdigit() else index + 1

    runs = [run_number(line, i) for i, line in enumerate(anno.read_text().splitlines())]
    mae = np.mean(np.abs(pred - true), axis=1)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))

    out_dir = Path(args.checkpoint).parent
    split = Path(args.test_anno).stem
    z_path = Path(args.anno_path) / split / "z.npy"
    z = np.load(z_path) if z_path.exists() else np.arange(true.shape[-1])
    np.savez(out_dir / f"{split}_predictions.npz", true=true, pred=pred, z=z, run=np.array(runs))
    with (out_dir / f"{split}_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "mae_kg_m3", "rmse_kg_m3"))
        writer.writerows(zip(runs, mae, rmse))

    print(f"{args.dataset} ({len(runs)} {split} samples): "
          f"MAE {mae.mean():.4f} kg/m^3, RMSE {np.sqrt(np.mean((pred - true) ** 2)):.4f} kg/m^3")
    print(f"Saved {out_dir / f'{split}_predictions.npz'} and {out_dir / f'{split}_metrics.csv'}")


if __name__ == "__main__":
    main()
