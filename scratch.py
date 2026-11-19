import json
import numpy as np

results = np.load("C:/Users/starb/Downloads/runs/bathysaurus/test_predictions.npz")
pred = results["predictions"]
target = results["targets"]

with open("C:/Users/starb/Downloads/runs/bathysaurus/preprocessing.json") as f:
    norm = json.load(f)

# Constant baseline: predict the training-set mean everywhere.
baseline = np.full_like(target, norm["target_mean"])

def metrics(p, y):
    error = p - y
    return np.mean(np.abs(error)), np.sqrt(np.mean(error ** 2))

model_mae, model_rmse = metrics(pred, target)
base_mae, base_rmse = metrics(baseline, target)

print(f"Model:    MAE={model_mae:.4f}, RMSE={model_rmse:.4f}")
print(f"Baseline: MAE={base_mae:.4f}, RMSE={base_rmse:.4f}")
print(f"RMSE relative to training target std: "
      f"{model_rmse / norm['target_std']:.3f}")