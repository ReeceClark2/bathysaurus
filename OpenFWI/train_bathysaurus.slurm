#!/bin/bash
# Train and evaluate InversionNet1D on one Bathysaurus dataset with OpenFWI.
# Usage (from the OpenFWI folder):
#   sbatch train_bathysaurus.slurm 07_flat_noisy_r4 dynamic
# All 12 datasets x both modes (24 jobs):
#   for d in $(ls ~/bathysaurus/data/datasets | grep -v _relative); do
#     for m in dynamic anomaly; do sbatch train_bathysaurus.slurm "$d" "$m"; done; done
#SBATCH --job-name=invnet1d
#SBATCH --partition=gpu              # <-- your cluster's GPU partition
#SBATCH --gres=gpu:1                 # OpenFWI's InversionNet trains on a single GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=invnet1d_%x_%j.out

set -e
module purge
module load anaconda
conda activate bathysaurus           # needs torch, torchvision, tensorboard
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DATASET=${1:?usage: sbatch train_bathysaurus.slurm <dataset folder name> <dynamic|anomaly>}
MODE=${2:?usage: sbatch train_bathysaurus.slurm <dataset folder name> <dynamic|anomaly>}
DATASETS_DIR=${DATASETS_DIR:-$HOME/bathysaurus/data/datasets}   # override with: DATASETS_DIR=... sbatch ...

# 200 epochs (4 blocks of 50), L1 loss only as in the OpenFWI README, batch 32.
python -u train.py -ds "${DATASET}-${MODE}" -m InversionNet1D -fs 1 \
    -ap "$DATASETS_DIR/$DATASET" -t train.txt -v validation.txt \
    -n "$DATASET" -s "$MODE" -b 32 --lr 1e-3 -eb 50 -nb 4 -j 4 -g2v 0 --tensorboard

python -u test_bathysaurus.py -ds "${DATASET}-${MODE}" -ap "$DATASETS_DIR/$DATASET" -t test.txt \
    -c "Invnet_models/$DATASET/$MODE/checkpoint.pth"
