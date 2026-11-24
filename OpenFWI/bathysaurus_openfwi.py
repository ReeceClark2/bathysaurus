"""Bathysaurus adapter for OpenFWI (https://github.com/lanl/OpenFWI).

Put this file in the OpenFWI folder. Importing it registers the model
"InversionNet1D" in network.model_dict, so train.py can use `-m InversionNet1D`.

InversionNet1D keeps InversionNet's design -- a convolutional encoder that
compresses the recordings to a single latent vector, then a transposed-
convolution decoder with a tanh output in [-1, 1] -- but for this problem:

    input   (N, 1, n_time, n_receivers)   receiver pressure time series
    output  (N, 1, n_out)                 density profile (n_out = Nz = 128)

Receivers are treated as input channels and convolution runs along time
(OpenFWI's InversionNet convolves over a 1000 x 70 shot gather instead).
The same model therefore works for 1, 2, 4 or 8 receivers.

Pressure is pre-processed by PressureTransform, then min-max scaled to
[-1, 1] like OpenFWI does:
    "dynamic": subtract each receiver's mean pressure before the impulse,
               leaving only the pressure change caused by the waves.
    "anomaly": subtract the fixed reference rho_ref * g * depth of each
               receiver; keeps the static pressure, which also depends on
               the density profile.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose

import network
import transforms as T


# ----------------------------------------------------------------- building blocks

class ConvBlock1d(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, relu_slop=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_fea, out_fea, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_fea),
            nn.LeakyReLU(relu_slop, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock1d(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_fea, out_fea, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_fea),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlock1d_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_fea, out_fea, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_fea),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


# ------------------------------------------------------------------------ model

class InversionNet1D(nn.Module):
    def __init__(self, n_receivers, n_time, n_out, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512,
                 sample_temporal=1, **kwargs):  # **kwargs absorbs train.py's upsample_mode, sample_spatial
        super().__init__()
        n_time = len(range(0, n_time, sample_temporal))  # FWIDataset keeps every sample_temporal-th sample
        self.n_out = n_out

        # Encoder: halve the time axis repeatedly (181 -> 91 -> 46 -> 23 -> 12 -> 6 -> 3)
        self.encoder = nn.Sequential(
            ConvBlock1d(n_receivers, dim1, kernel_size=7, stride=2, padding=3),
            ConvBlock1d(dim1, dim2, stride=2), ConvBlock1d(dim2, dim2),
            ConvBlock1d(dim2, dim2, stride=2), ConvBlock1d(dim2, dim2),
            ConvBlock1d(dim2, dim3, stride=2), ConvBlock1d(dim3, dim3),
            ConvBlock1d(dim3, dim4, stride=2), ConvBlock1d(dim4, dim4),
            ConvBlock1d(dim4, dim4, stride=2), ConvBlock1d(dim4, dim4),
        )
        with torch.no_grad():
            remaining = self.encoder.eval()(torch.zeros(2, n_receivers, n_time)).shape[-1]
        self.encoder.train()
        # As in InversionNet: one convolution over the whole remaining length -> latent vector
        self.latent = ConvBlock1d(dim4, dim5, kernel_size=remaining, padding=0)

        # Decoder: grow to start * 2^5 >= n_out (4 * 32 = 128), then crop to n_out
        start = -(-n_out // 32)
        self.decoder = nn.Sequential(
            DeconvBlock1d(dim5, dim5, kernel_size=start, stride=1, padding=0), ConvBlock1d(dim5, dim5),
            DeconvBlock1d(dim5, dim4), ConvBlock1d(dim4, dim4),
            DeconvBlock1d(dim4, dim3), ConvBlock1d(dim3, dim3),
            DeconvBlock1d(dim3, dim2), ConvBlock1d(dim2, dim2),
            DeconvBlock1d(dim2, dim1), ConvBlock1d(dim1, dim1),
            DeconvBlock1d(dim1, dim1), ConvBlock1d(dim1, dim1),
        )
        self.output = ConvBlock1d_Tanh(dim1, 1)

    def forward(self, x):
        x = x[:, 0].transpose(1, 2)          # (N, 1, T, R) -> (N, R, T): receivers as channels
        x = self.latent(self.encoder(x))      # (N, dim5, 1)
        x = self.decoder(x)                   # (N, dim1, start * 32)
        excess = x.shape[-1] - self.n_out
        x = x[..., excess // 2: excess // 2 + self.n_out]
        return self.output(x)                 # (N, 1, n_out)


network.model_dict["InversionNet1D"] = InversionNet1D


# ------------------------------------------------------------------- transforms

class PressureTransform:
    """Removes the large static pressure so the network sees Pa-scale signals."""

    def __init__(self, mode, reference_pa=None, baseline_samples=None):
        if mode not in ("dynamic", "anomaly"):
            raise ValueError('pressure mode must be "dynamic" or "anomaly"')
        self.mode = mode
        self.reference = None if reference_pa is None else np.asarray(reference_pa, dtype=np.float64)
        self.baseline_samples = baseline_samples

    def __call__(self, data):  # data: (1, T, R)
        data = np.asarray(data, dtype=np.float64)
        if self.mode == "dynamic":
            data = data - data[:, :self.baseline_samples, :].mean(axis=1, keepdims=True)
        else:
            data = data - self.reference[None, None, :]
        return data.astype(np.float32)


class CopyArray:
    """OpenFWI's MinMaxNormalize works in place; with preload=True that would re-normalize the
    stored labels every epoch. Copy first so the preloaded arrays are never modified."""

    def __call__(self, x):
        return np.array(x, dtype=np.float32, copy=True)


def is_bathysaurus(ctx):
    return "pressure_mode" in ctx


def build_transforms(ctx, sample_temporal=1):
    """Data and label transforms for a dataset_config.json entry written by prepare_openfwi.py."""
    baseline = max(1, -(-int(ctx["baseline_samples"]) // sample_temporal))
    transform_data = Compose([
        PressureTransform(ctx["pressure_mode"], ctx.get("reference_pa"), baseline),
        T.MinMaxNormalize(ctx["data_min"], ctx["data_max"]),
    ])
    transform_label = Compose([CopyArray(), T.MinMaxNormalize(ctx["label_min"], ctx["label_max"])])
    return transform_data, transform_label


def model_kwargs(ctx):
    return {"n_receivers": int(ctx["ng"]), "n_time": int(ctx["nt"]), "n_out": int(ctx["n_grid"])}
