from pathlib import Path
import random
import tomllib

import argparse
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Run this script from the project directory containing data/.
DATA_ROOT = Path("data")
BATCH_SIZE = 8
NUM_WORKERS = 0
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for more reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BathysaurusDataset(Dataset):
    """Read one pressure/target pair per line of a split list."""

    def __init__(self, pair_list: Path):
        self.pair_list = Path(pair_list)
        self.pairs = []

        if not self.pair_list.is_file():
            raise FileNotFoundError(f"Pair list not found: {self.pair_list}")

        for line_number, line in enumerate(
            self.pair_list.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"{self.pair_list}:{line_number}: expected two tab-separated "
                    "paths (data file and model file)"
                )

            data_path, model_path = (Path(field.strip()) for field in fields)

            # Pair-list paths are normally absolute. For relative paths, first
            # try the working directory, then the pair list's directory.
            if not data_path.is_absolute() and not data_path.exists():
                data_path = self.pair_list.parent / data_path
            if not model_path.is_absolute() and not model_path.exists():
                model_path = self.pair_list.parent / model_path

            self.pairs.append((data_path, model_path))

        if not self.pairs:
            raise ValueError(f"No data pairs found in {self.pair_list}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        data_path, model_path = self.pairs[index]

        # Stored shapes: (1, 1, Nt, Nr) and (1, 1, Nz).
        # Remove only the leading example axis; keep the channel axis.
        x = np.load(data_path).astype(np.float32, copy=False)
        y = np.load(model_path).astype(np.float32, copy=False)

        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Unexpected input shape in {data_path}: {x.shape}")
        if y.ndim != 3 or y.shape[0] != 1:
            raise ValueError(f"Unexpected target shape in {model_path}: {y.shape}")

        x = np.ascontiguousarray(x[0])  # (1, Nt, Nr)
        y = np.ascontiguousarray(y[0])  # (1, Nz)

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values found in {data_path}")
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite values found in {model_path}")

        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders():
    """Create loaders for the existing train/validation/test pair lists."""
    train_set = BathysaurusDataset(DATA_ROOT / "train.txt")
    validation_set = BathysaurusDataset(DATA_ROOT / "validation.txt")
    test_set = BathysaurusDataset(DATA_ROOT / "test.txt")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader, test_loader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InversionNetProfile(nn.Module):
    """Convolutional trace encoder with a 1D density-profile output."""

    def __init__(self, nz: int):
        super().__init__()
        self.nz = nz

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            # Allows the dense output layer to work with different Nt and Nr.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.profile_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 512),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, nz),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input shape (B, 1, Nt, Nr), got {tuple(x.shape)}"
            )

        features = self.encoder(x)
        profile = self.profile_head(features)

        # Retain the single density-channel dimension.
        return profile.unsqueeze(1)  # (B, 1, Nz)


# ---------------------------------------------------------------------------
# Training-only normalization
# ---------------------------------------------------------------------------

class ScalarMoments:
    """Numerically stable running mean and variance for scalar normalization."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return

        batch_count = values.size
        batch_mean = float(values.mean())
        batch_m2 = float(np.sum((values - batch_mean) ** 2))

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.m2 += (
            batch_m2
            + delta * delta * self.count * batch_count / total_count
        )
        self.count = total_count

    def result(self):
        if self.count == 0:
            raise ValueError("Cannot calculate statistics from an empty dataset.")

        variance = self.m2 / self.count
        std = float(np.sqrt(max(variance, 0.0)))
        return float(self.mean), max(std, 1e-8)


def fit_normalization(train_dataset):
    """Fit scalar input/target normalization using training data only."""
    x_moments = ScalarMoments()
    y_moments = ScalarMoments()

    for index in range(len(train_dataset)):
        x, y = train_dataset[index]
        x_moments.update(x.numpy())
        y_moments.update(y.numpy())

    x_mean, x_std = x_moments.result()
    y_mean, y_std = y_moments.result()

    return {
        "input_mean": x_mean,
        "input_std": x_std,
        "target_mean": y_mean,
        "target_std": y_std,
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, normalization, training):
    """Run one training or validation epoch and return mean normalized MSE."""
    model.train(training)
    total_squared_error = 0.0
    total_values = 0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        x = (x - normalization["input_mean"]) / normalization["input_std"]
        y_normalized = (
            y - normalization["target_mean"]
        ) / normalization["target_std"]

        with torch.set_grad_enabled(training):
            prediction = model(x)

            if prediction.shape != y_normalized.shape:
                raise RuntimeError(
                    "Prediction/target shape mismatch: "
                    f"prediction={tuple(prediction.shape)}, "
                    f"target={tuple(y_normalized.shape)}"
                )

            loss = torch.mean((prediction - y_normalized) ** 2)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        n_values = y_normalized.numel()
        total_squared_error += loss.detach().item() * n_values
        total_values += n_values

    if total_values == 0:
        raise ValueError("The data loader produced no samples.")

    return total_squared_error / total_values


@torch.no_grad()
def predict_profiles(model, loader, device, normalization):
    """Return predictions and targets in original density units."""
    model.eval()
    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        x = (x - normalization["input_mean"]) / normalization["input_std"]

        prediction_normalized = model(x)
        prediction = (
            prediction_normalized * normalization["target_std"]
            + normalization["target_mean"]
        )

        if prediction.shape != y.shape:
            raise RuntimeError(
                "Prediction/target shape mismatch during evaluation: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(y.shape)}"
            )

        predictions.append(prediction.cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def check_split_coordinates(data_root: Path) -> None:
    """Check available time/depth coordinate files agree across splits."""
    for coordinate_name in ("time.npy", "z.npy"):
        found = []

        for split in ("train", "validation", "test"):
            path = data_root / split / coordinate_name
            if path.is_file():
                found.append((split, np.load(path)))

        if len(found) < 2:
            continue

        reference_split, reference = found[0]
        for split, coordinate in found[1:]:
            if reference.shape != coordinate.shape or not np.allclose(
                reference, coordinate, rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"{coordinate_name} differs between {reference_split} "
                    f"and {split} splits."
                )


def main():
    config_path = Path("config.toml")
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    data_root = Path(config.get("data_root", "data"))
    output_dir = Path(config.get("output_dir", "runs/bathysaurus"))
    batch_size = int(config.get("batch_size", BATCH_SIZE))
    epochs = int(config.get("epochs", 100))
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-5))
    seed = int(config.get("seed", SEED))
    num_workers = int(config.get("num_workers", NUM_WORKERS))

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_split_coordinates(data_root)

    train_set = BathysaurusDataset(data_root / "train.txt")
    validation_set = BathysaurusDataset(data_root / "validation.txt")
    test_set = BathysaurusDataset(data_root / "test.txt")

    pin_memory = torch.cuda.is_available()
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_set, shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        test_set, shuffle=False, **loader_options
    )

    # Verify the actual file shapes before starting training.
    x_example, y_example = train_set[0]
    print(f"One input:  shape={tuple(x_example.shape)}, dtype={x_example.dtype}")
    print(f"One target: shape={tuple(y_example.shape)}, dtype={y_example.dtype}")

    first_x, first_y = next(iter(train_loader))
    print(f"Input batch:  {tuple(first_x.shape)}")
    print(f"Target batch: {tuple(first_y.shape)}")

    if first_x.ndim != 4 or first_x.shape[1] != 1:
        raise ValueError(
            f"Expected batch inputs shaped (B, 1, Nt, Nr), got {tuple(first_x.shape)}"
        )
    if first_y.ndim != 3 or first_y.shape[1] != 1:
        raise ValueError(
            f"Expected batch targets shaped (B, 1, Nz), got {tuple(first_y.shape)}"
        )

    nz = int(y_example.shape[-1])
    normalization = fit_normalization(train_set)

    with (output_dir / "preprocessing.json").open("w") as f:
        json.dump(normalization, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training examples: {len(train_set)}")
    print(f"Validation examples: {len(validation_set)}")
    print(f"Test examples: {len(test_set)}")
    print(f"Profile length Nz: {nz}")
    print(f"Training-only normalization: {normalization}")

    model = InversionNetProfile(nz=nz).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_loss = float("inf")
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            normalization=normalization,
            training=True,
        )
        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=None,
            device=device,
            normalization=normalization,
            training=False,
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train MSE (normalized): {train_loss:.6g} | "
            f"validation MSE (normalized): {validation_loss:.6g}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "nz": nz,
                    "normalization": normalization,
                    "epoch": epoch,
                    "validation_loss_normalized": validation_loss,
                    "seed": seed,
                },
                checkpoint_path,
            )

    # Load the checkpoint selected by validation loss, then evaluate test once.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions, targets = predict_profiles(
        model=model,
        loader=test_loader,
        device=device,
        normalization=normalization,
    )

    # Arrays are (N, 1, Nz); remove only the single density channel.
    predictions = predictions[:, 0, :]
    targets = targets[:, 0, :]

    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    print(f"Selected checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Best validation MSE (normalized): "
        f"{checkpoint['validation_loss_normalized']:.6g}"
    )
    print(f"Test MAE:  {mae:.6g}")
    print(f"Test RMSE: {rmse:.6g}")

    # Save predictions and targets in original density units.
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        predictions=predictions,
        targets=targets,
        errors=errors,
    )

    # Calculate and save mean absolute error at each depth.
    mae_by_depth = np.mean(np.abs(errors), axis=0)
    z_path = data_root / "test" / "z.npy"

    if z_path.is_file():
        z = np.load(z_path)
        if z.size != mae_by_depth.size:
            raise ValueError(
                f"Depth coordinate has {z.size} values, but profiles have "
                f"{mae_by_depth.size} depth cells."
            )
        depth_results = np.column_stack((z.ravel(), mae_by_depth))
        header = "depth,mae"
    else:
        depth_results = np.column_stack(
            (np.arange(mae_by_depth.size), mae_by_depth)
        )
        header = "depth_index,mae"

    np.savetxt(
        output_dir / "test_mae_by_depth.csv",
        depth_results,
        delimiter=",",
        header=header,
        comments="",
    )

    metrics = {
        "selected_epoch": int(checkpoint["epoch"]),
        "best_validation_mse_normalized": float(
            checkpoint["validation_loss_normalized"]
        ),
        "test_mae": mae,
        "test_rmse": rmse,
        "num_test_examples": int(targets.shape[0]),
        "profile_length": int(targets.shape[1]),
        "seed": seed,
    }

    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved test predictions: {output_dir / 'test_predictions.npz'}")
    print(f"Saved depth-wise MAE: {output_dir / 'test_mae_by_depth.csv'}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
