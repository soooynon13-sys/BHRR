#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for restoring GCM(tasmax_raw) / tmax_vit -> tmax_obs using Restormer (64x64 patches)

Options:
    - If use_gcm_input = True:
        * Input: tasmax_raw.nc (variable gcm_var, e.g., "tasmax_raw")
        * Target: tmax_obs in tmax_all_methods_comparison.nc
        * tasmax_raw is interpolated onto the time/lat/lon grid of tmax_obs using xarray.interp

    - If use_gcm_input = False:
        * Input: input_var in tmax_all_methods_comparison.nc (e.g., tmax_vit)
        * Target: tmax_obs

Common settings:
- Original target dimensions: (time, lat, lon) ≈ (12784, 200, 280)
- Training/validation: 64x64 patches
- Every epoch:
    * Save PNGs of 64x64 validation patches
    * Perform sliding-window inference over the full 200x280 domain using 64x64 tiles and save PNG
    * Compute PSNR/SSIM on the full-domain inference result and save logs/CSV
    * Also record baseline (input vs target) PSNR/SSIM in the same CSV

Requirements:
    pip install xarray netCDF4 tqdm matplotlib
    + Put the original Restormer code (restormer_arch.py) in the same directory
      so that it is importable with: from restormer_arch import Restormer
"""

import os
import math
import random
from types import SimpleNamespace
import csv  # CSV logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# matplotlib for saving figures (use Agg backend to work without X)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Restormer core is imported from the GitHub file
from restormer_arch import Restormer   # <- adjust to match file/class name


# ---------------------------------------------------------
# 0. Utilities
# ---------------------------------------------------------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_device():
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"[CUDA] {dev}, free={free/1e9:.2f} GB / total={total/1e9:.2f} GB")
    else:
        print("[CUDA] Not available, using CPU.")


def _fill_nans_robust(arr: np.ndarray) -> np.ndarray:
    """
    Safely fill NaNs in an array with shape [T, H, W].

    - First fill using the time-wise mean for each time step.
    - If NaNs still remain, fill them with the overall global mean.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isnan(arr).any():
        return arr

    # Time-wise mean
    m_t = np.nanmean(arr, axis=(1, 2), keepdims=True)  # [T,1,1]
    global_mean = np.nanmean(arr)
    if np.isnan(global_mean):
        global_mean = 0.0

    m_t = np.where(np.isnan(m_t), global_mean, m_t)
    arr = np.where(np.isnan(arr), m_t, arr)
    arr = np.nan_to_num(arr, nan=float(global_mean))
    return arr


# ---------------------------------------------------------
# 1. NetCDF loading
# ---------------------------------------------------------
def _try_import_xarray():
    try:
        import xarray as xr
        return xr
    except Exception:
        raise RuntimeError("xarray is required. Please run `pip install xarray netCDF4` first.")


def load_tmax_nc(
    nc_path: str,
    input_var: str = "tmax_vit",
    target_var: str = "tmax_obs",
):
    """
    Load input/target variables with shape (time, lat, lon)
    from the same NetCDF file.
    """
    xr = _try_import_xarray()
    ds = xr.open_dataset(nc_path)

    if input_var not in ds or target_var not in ds:
        raise KeyError(
            f"Variable '{input_var}' or '{target_var}' not found in '{nc_path}'. "
            f"Available variables: {list(ds.data_vars)}"
        )

    x_da = ds[input_var]   # (time, lat, lon)
    y_da = ds[target_var]  # (time, lat, lon)

    # Keep original names for time, lat, lon and only use values
    time_vals = x_da["time"].values if "time" in x_da.dims else np.arange(x_da.shape[0])
    lat_vals = x_da["lat"].values if "lat" in x_da.dims else np.arange(x_da.shape[1])
    lon_vals = x_da["lon"].values if "lon" in x_da.dims else np.arange(x_da.shape[2])

    x = x_da.values.astype(np.float32)  # [T, H, W]
    y = y_da.values.astype(np.float32)  # [T, H, W]

    ds.close()

    print(f"[NC] '{nc_path}' loaded: {input_var} = {x.shape}, {target_var} = {y.shape}")

    # Robust NaN filling
    x = _fill_nans_robust(x)
    y = _fill_nans_robust(y)

    return x, y, time_vals, lat_vals, lon_vals


def _guess_dim_renames(da):
    """
    Guess dimension names for time/lat/lon in a GCM variable.

    Returns a rename dict mapping original dim names to 'time', 'lat', or 'lon',
    e.g., {'latitude': 'lat', 'longitude': 'lon'}.
    """
    rename = {}
    dims = list(da.dims)
    lower_map = {d.lower(): d for d in dims}

    # time
    if "time" not in dims:
        for key, orig in lower_map.items():
            if "time" in key:
                rename[orig] = "time"
                break

    # lat
    if "lat" not in dims:
        for key, orig in lower_map.items():
            if "lat" in key or "y" == key:
                rename[orig] = "lat"
                break

    # lon
    if "lon" not in dims:
        for key, orig in lower_map.items():
            if "lon" in key or "x" == key:
                rename[orig] = "lon"
                break

    return rename


def load_gcm_regridded_to_obs(
    obs_nc_path: str,
    target_var: str = "tmax_obs",
    gcm_nc_path: str = "tasmax_raw.nc",
    gcm_var: str = "tasmax_raw",
):
    """
    (1) Read target_var=tmax_obs from an obs NetCDF file (e.g., tmax_all_methods_comparison.nc)
    (2) Read gcm_var from a GCM NetCDF file (e.g., tasmax_raw.nc)
    (3) Interpolate the GCM field onto the time/lat/lon grid of tmax_obs using xarray.interp

    Returns:
        x, y, time_vals, lat_vals, lon_vals  (all on the obs grid)
    """
    xr = _try_import_xarray()

    # --- Target (obs) ---
    ds_obs = xr.open_dataset(obs_nc_path)
    if target_var not in ds_obs:
        raise KeyError(
            f"Variable '{target_var}' not found in '{obs_nc_path}'. "
            f"Available variables: {list(ds_obs.data_vars)}"
        )
    y_da = ds_obs[target_var]  # (time, lat, lon)

    # Obs coordinates
    if not all(dim in y_da.dims for dim in ("time", "lat", "lon")):
        raise ValueError(f"Target variable '{target_var}' does not have dims (time, lat, lon): {y_da.dims}")

    time_obs = y_da["time"]
    lat_obs = y_da["lat"]
    lon_obs = y_da["lon"]

    # --- GCM ---
    ds_gcm = xr.open_dataset(gcm_nc_path)
    if gcm_var not in ds_gcm:
        raise KeyError(
            f"Variable '{gcm_var}' not found in '{gcm_nc_path}'. "
            f"Available variables: {list(ds_gcm.data_vars)}"
        )

    gcm_da = ds_gcm[gcm_var]

    # Rename dimensions to time/lat/lon (e.g., latitude, longitude → lat, lon)
    rename = _guess_dim_renames(gcm_da)
    if rename:
        print(f"[GCM] rename dims: {rename}")
        ds_gcm = ds_gcm.rename(rename)
        gcm_da = ds_gcm[gcm_var]

    for dim in ("time", "lat", "lon"):
        if dim not in gcm_da.dims:
            raise ValueError(
                f"Dimension '{dim}' not found in GCM variable '{gcm_var}'. dims={gcm_da.dims}"
            )

    # Sort in time/space
    gcm_da = gcm_da.sortby("time")
    y_da = y_da.sortby("time")

    # --- Interpolate onto obs grid ---
    print("[GCM] Interpolating to obs grid (time, lat, lon)...")
    gcm_interp = gcm_da.interp(
        time=time_obs,
        lat=lat_obs,
        lon=lon_obs,
        method="linear",
    )

    x = gcm_interp.values.astype(np.float32)  # [T,H,W]
    y = y_da.values.astype(np.float32)        # [T,H,W]

    ds_obs.close()
    ds_gcm.close()

    print(
        f"[NC] GCM '{gcm_nc_path}'({gcm_var}) regridded to obs grid: "
        f"x={x.shape}, y={y.shape}"
    )

    x = _fill_nans_robust(x)
    y = _fill_nans_robust(y)

    time_vals = time_obs.values
    lat_vals = lat_obs.values
    lon_vals = lon_obs.values

    return x, y, time_vals, lat_vals, lon_vals


# ---------------------------------------------------------
# 2. Dataset (patch-based 64x64)
# ---------------------------------------------------------
class TmaxRestorationDataset(Dataset):
    """
    Dataset for tmax_* -> tmax_obs restoration.

    - patch_size=None: use the full image (1, H, W)
    - patch_size=64: randomly crop (1, 64, 64) patches
    - Normalize with global mean/standard deviation
    """
    def __init__(
        self,
        x_all: np.ndarray,  # [T, H, W]
        y_all: np.ndarray,  # [T, H, W]
        indices: np.ndarray,
        stats: dict,
        augment: bool = True,
        normalize: bool = True,
        patch_size: int = None,
        random_crop: bool = True,
    ):
        super().__init__()
        self.x_all = x_all
        self.y_all = y_all
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.normalize = normalize
        self.mean = float(stats["mean"])
        self.std = float(stats["std"] + 1e-8)

        self.patch_size = patch_size
        self.random_crop = random_crop

    def __len__(self):
        return len(self.indices)

    def _augment(self, x: torch.Tensor, y: torch.Tensor):
        # Random horizontal/vertical flips and transpose (lat/lon swap)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        # For 64x64 patches, transpose keeps shape identical
        if random.random() < 0.25:
            x = x.transpose(1, 2)
            y = y.transpose(1, 2)
        return x, y

    def _crop_patch(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: [1, H, W]
        If patch_size is None, use the full image. Otherwise crop a patch of size (patch_size, patch_size).
        """
        if self.patch_size is None:
            return x, y

        _, H, W = x.shape
        ps = self.patch_size

        if H < ps or W < ps:
            # If the image is smaller than the patch size, just use the full image (rare case)
            return x, y

        if self.random_crop:
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
        else:
            # Center crop (used for validation)
            top = (H - ps) // 2
            left = (W - ps) // 2

        x = x[:, top:top+ps, left:left+ps]
        y = y[:, top:top+ps, left:left+ps]
        return x, y

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        x = self.x_all[t]  # [H, W]
        y = self.y_all[t]  # [H, W]

        x = torch.from_numpy(x).unsqueeze(0)  # [1, H, W]
        y = torch.from_numpy(y).unsqueeze(0)  # [1, H, W]

        # Safety: remove any remaining NaN/Inf
        x = torch.nan_to_num(x, nan=self.mean, posinf=self.mean, neginf=self.mean)
        y = torch.nan_to_num(y, nan=self.mean, posinf=self.mean, neginf=self.mean)

        # Patch crop (train: 64x64, validation: 64x64 center patch if val_full_image=False)
        x, y = self._crop_patch(x, y)

        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        if self.augment:
            x, y = self._augment(x, y)

        return x, y


def build_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    patch_size: int = 64,         # Training patch size (64x64)
    val_full_image: bool = False, # If True, use full 200x280 images for validation (otherwise 64x64)
    val_batch_size: int = 32,
):
    """
    Build training and validation DataLoaders from x, y with shape [T, H, W].
    """
    T = x.shape[0]
    indices = np.arange(T)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(T * val_ratio)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Compute statistics on training time steps
    mean = float(y[train_idx].mean())
    std = float(y[train_idx].std() + 1e-8)
    stats = {"mean": mean, "std": std}
    print(f"[STATS] train target mean={mean:.4f}, std={std:.4f}")

    # === Train: random 64x64 patches + augmentation ===
    train_ds = TmaxRestorationDataset(
        x_all=x,
        y_all=y,
        indices=train_idx,
        stats=stats,
        augment=True,
        normalize=True,
        patch_size=patch_size,
        random_crop=True,
    )

    # === Validation: center 64x64 patches or full images ===
    if val_full_image:
        # Use full 200x280 images
        val_ds = TmaxRestorationDataset(
            x_all=x,
            y_all=y,
            indices=val_idx,
            stats=stats,
            augment=False,
            normalize=True,
            patch_size=None,      # use full image
            random_crop=False,
        )
    else:
        # 64x64 center patches for validation
        val_ds = TmaxRestorationDataset(
            x_all=x,
            y_all=y,
            indices=val_idx,
            stats=stats,
            augment=False,
            normalize=True,
            patch_size=patch_size,
            random_crop=False,    # center crop
        )

    bs_val = val_batch_size

    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    return train_loader, val_loader, stats


# ---------------------------------------------------------
# 3. SSIM & Loss
# ---------------------------------------------------------
class SSIMLoss(nn.Module):
    """
    SSIM loss (1 - SSIM) computed in normalized space.
    """
    def __init__(self, win_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.win_size = win_size
        self.C1 = C1
        self.C2 = C2
        self.register_buffer(
            "window",
            torch.ones(1, 1, win_size, win_size) / (win_size * win_size)
        )

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        B, C, H, W = x.shape
        w = self.window.to(x.device, x.dtype)
        w = w.expand(C, 1, self.win_size, self.win_size)
        pad = self.win_size // 2

        mu_x = F.conv2d(x, w, padding=pad, groups=C)
        mu_y = F.conv2d(y, w, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x2
        sigma_y = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x2 + mu_y2 + self.C1) * (sigma_x + sigma_y + self.C2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class RestorationLoss(nn.Module):
    """
    Combination of L1 + MSE + SSIM losses (weights can be adjusted).
    """
    def __init__(self, w_l1=1.0, w_mse=0.0, w_ssim=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.w_l1 = w_l1
        self.w_mse = w_mse
        self.w_ssim = w_ssim

    def forward(self, pred, target):
        loss = 0.0
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * self.l1(pred, target)
        if self.w_mse > 0:
            loss = loss + self.w_mse * self.mse(pred, target)
        if self.w_ssim > 0:
            loss = loss + self.w_ssim * self.ssim(pred, target)
        return loss


# ---------------------------------------------------------
# 3.5. Numpy-based PSNR/SSIM helpers (for full-domain inference)
# ---------------------------------------------------------
def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute PSNR on numpy 2D images (pred, target) in physical units.

    The data range is taken as (max(target) - min(target)).
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    data_range = float(target.max() - target.min())
    if data_range <= 1e-12:
        data_range = 1.0
    psnr = 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)
    return float(psnr)


def ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute SSIM on numpy 2D images (pred, target).

    - Normalize to [0,1] using target's min and max, then use the SSIMLoss defined above.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    min_t = float(target.min())
    max_t = float(target.max())
    scale = max_t - min_t
    if scale <= 1e-12:
        scale = 1.0

    pred_norm = (pred - min_t) / scale
    target_norm = (target - min_t) / scale

    pred_t = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    target_t = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0)

    loss_fn = SSIMLoss()
    with torch.no_grad():
        loss_val = float(loss_fn(pred_t, target_t))  # 1 - SSIM
    ssim_val = 1.0 - loss_val
    return float(ssim_val)


# ---------------------------------------------------------
# 4. Model builder (Restormer)
# ---------------------------------------------------------
def build_restormer(
    inp_channels=1,
    out_channels=1,
    dim=32,                          # small model to save memory
    num_blocks=(3, 3, 3, 3),
    num_refinement_blocks=3,
    heads=(1, 2, 4, 8),
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="WithBias",
):
    """
    Build Restormer with a signature compatible with restormer_arch.py.

    (Assumes the official implementation or a compatible variant.)
    """
    model = Restormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=list(num_blocks),
        num_refinement_blocks=num_refinement_blocks,
        heads=list(heads),
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        dual_pixel_task=False,
    )
    print(f"[MODEL] Restormer params = {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


# ---------------------------------------------------------
# 4.5. Sliding-tile full-image inference (200x280)
# ---------------------------------------------------------
@torch.no_grad()
def predict_full_image_tiled(
    model: nn.Module,
    x_full: torch.Tensor,   # [1, H, W] or [B=1, 1, H, W]
    tile_size: int = 64,
    overlap: int = 16,
    device: str = "cuda:0",
):
    """
    Perform inference on a full 200x280 image by tiling 64x64 patches.

    - x_full: [1, H, W] (normalized input single image)
    - Returns: [1, H, W] (model output in normalized scale)
    """
    model.eval()

    if x_full.ndim == 3:
        x_full = x_full.unsqueeze(0)  # [1, 1, H, W]
    assert x_full.ndim == 4 and x_full.shape[0] == 1

    x_full = x_full.to(device)
    _, _, H, W = x_full.shape

    tile = tile_size
    stride = tile_size - overlap

    # Output/weight buffers
    out = torch.zeros_like(x_full)
    weight = torch.zeros_like(x_full)

    # Tile start positions
    ys = list(range(0, max(H - tile + 1, 1), stride))
    xs = list(range(0, max(W - tile + 1, 1), stride))

    # Ensure edges are fully covered by adding final tiles
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)

    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for top in ys:
        for left in xs:
            top = int(top)
            left = int(left)

            patch = x_full[:, :, top:top+tile, left:left+tile]  # [1,1,64,64]
            if patch.shape[-2] != tile or patch.shape[-1] != tile:
                continue

            pred_patch = model(patch)  # [1,1,64,64]
            pred_patch = pred_patch.detach()

            out[:, :, top:top+tile, left:left+tile] += pred_patch
            weight[:, :, top:top+tile, left:left+tile] += 1.0

    # Avoid division by zero
    weight = torch.clamp(weight, min=1.0)
    out = out / weight

    # [1,1,H,W] -> [1,H,W]
    return out.squeeze(0)  # [1, H, W]


# ---------------------------------------------------------
# 5. Trainer
# ---------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stats: dict,
        device: str = "cuda:0",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 150,
        out_dir: str = "restormer_ckpt",
        use_amp: bool = True,
        # Example full-domain image (200x280) for inference
        full_x_example: np.ndarray = None,  # [H, W]
        full_y_example: np.ndarray = None,  # [H, W]
        tile_size: int = 64,
        tile_overlap: int = 16,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.out_dir = out_dir
        self.stats = stats  # mean, std (normalization info)
        ensure_dir(out_dir)
        ensure_dir(os.path.join(out_dir, "epoch_png"))
        ensure_dir(os.path.join(out_dir, "epoch_png_full"))

        self.criterion = RestorationLoss(w_l1=1.0, w_mse=0.0, w_ssim=0.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        self.best_val = float("inf")

        # Example full-domain images (if provided)
        self.full_x_example = full_x_example
        self.full_y_example = full_y_example
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # === Baseline PSNR/SSIM (input vs target, physical units) ===
        self.baseline_psnr = None
        self.baseline_ssim = None
        if (self.full_x_example is not None) and (self.full_y_example is not None):
            try:
                self.baseline_psnr = psnr_np(self.full_x_example, self.full_y_example)
                self.baseline_ssim = ssim_np(self.full_x_example, self.full_y_example)
                print(
                    "[BASELINE] Input vs Target | "
                    f"PSNR={self.baseline_psnr:.3f} dB | SSIM={self.baseline_ssim:.4f}"
                )
            except Exception as e:
                print(f"[BASELINE] Failed to compute baseline PSNR/SSIM: {e}")

        # Initialize CSV file for full-domain PSNR/SSIM
        self.full_metrics_csv = os.path.join(self.out_dir, "full_metrics.csv")
        if (not os.path.exists(self.full_metrics_csv)) or os.path.getsize(self.full_metrics_csv) == 0:
            with open(self.full_metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                # Store epoch-wise model performance plus baseline
                writer.writerow(["epoch", "full_psnr", "full_ssim",
                                 "baseline_psnr", "baseline_ssim"])

    @staticmethod
    def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        PSNR in normalized space (mean 0, std 1).
        Assumes data_range = 1, so this is a relative metric.
        """
        mse = F.mse_loss(pred, target).item()
        if mse <= 1e-12:
            return 99.0
        data_range = 1.0
        psnr = 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
        return float(psnr)

    def _save_epoch_png(
        self,
        epoch: int,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        max_samples: int = 1,
    ):
        """
        Save 64x64 patch PNGs for the first validation batch at each epoch.

        x, y, pred are normalized tensors with shape [B,1,H,W].
        """
        mean = self.stats["mean"]
        std = self.stats["std"]

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()

        B = x_np.shape[0]
        n_samples = min(max_samples, B)

        for i in range(n_samples):
            in_img = x_np[i, 0] * std + mean   # physical units
            tgt_img = y_np[i, 0] * std + mean  # physical units
            prd_img = p_np[i, 0] * std + mean  # physical units

            vmin = min(in_img.min(), tgt_img.min(), prd_img.min())
            vmax = max(in_img.max(), tgt_img.max(), prd_img.max())

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            im0 = axs[0].imshow(in_img, vmin=vmin, vmax=vmax)
            axs[0].set_title("Input")
            axs[0].axis("off")

            im1 = axs[1].imshow(tgt_img, vmin=vmin, vmax=vmax)
            axs[1].set_title("Target (tmax_obs)")
            axs[1].axis("off")

            im2 = axs[2].imshow(prd_img, vmin=vmin, vmax=vmax)
            axs[2].set_title("Predicted (64x64)")
            axs[2].axis("off")

            fig.colorbar(im2, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
            plt.tight_layout()

            fname = os.path.join(
                self.out_dir,
                "epoch_png",
                f"epoch{epoch:03d}_patch_sample{i:02d}.png"
            )
            plt.savefig(fname, dpi=150)
            plt.close(fig)

    def _save_full_image_tiled(self, epoch: int):
        """
        Run tiled inference over the full 200x280 domain and save a PNG.

        full_x_example, full_y_example are numpy arrays [H, W] in physical units.
        Also compute PSNR/SSIM in physical units and log to CSV.
        """
        if self.full_x_example is None or self.full_y_example is None:
            return

        mean = self.stats["mean"]
        std = self.stats["std"]

        # [H,W] -> [1,H,W] tensor, then normalize
        x_full = torch.from_numpy(self.full_x_example).float().unsqueeze(0)  # [1,H,W]
        y_full = torch.from_numpy(self.full_y_example).float().unsqueeze(0)  # [1,H,W]

        x_full_norm = (x_full - mean) / std  # [1,H,W]

        # Tiled inference
        pred_norm = predict_full_image_tiled(
            self.model,
            x_full_norm,          # [1,H,W]
            tile_size=self.tile_size,
            overlap=self.tile_overlap,
            device=self.device,
        )  # [1,H,W]

        # Inverse normalization to physical units
        in_img = x_full.numpy()[0]              # physical units
        tgt_img = y_full.numpy()[0]             # physical units
        prd_img = (pred_norm * std + mean).cpu().numpy()[0]  # physical units

        # === PSNR/SSIM (physical units) ===
        full_psnr = psnr_np(prd_img, tgt_img)
        full_ssim = ssim_np(prd_img, tgt_img)

        # Print with baseline if available
        if (self.baseline_psnr is not None) and (self.baseline_ssim is not None):
            print(
                f"[FULL] Epoch {epoch:03d} | PSNR={full_psnr:.3f} dB | SSIM={full_ssim:.4f} "
                f"| BASE_PSNR={self.baseline_psnr:.3f} dB | BASE_SSIM={self.baseline_ssim:.4f}"
            )
        else:
            print(f"[FULL] Epoch {epoch:03d} | PSNR={full_psnr:.3f} dB | SSIM={full_ssim:.4f}")

        # Append to CSV (epoch, full_psnr, full_ssim, baseline_psnr, baseline_ssim)
        with open(self.full_metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            b_psnr = float(self.baseline_psnr) if self.baseline_psnr is not None else ""
            b_ssim = float(self.baseline_ssim) if self.baseline_ssim is not None else ""
            writer.writerow([int(epoch), float(full_psnr), float(full_ssim), b_psnr, b_ssim])

        # === Save figure ===
        vmin = min(in_img.min(), tgt_img.min(), prd_img.min())
        vmax = max(in_img.max(), tgt_img.max(), prd_img.max())

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        im0 = axs[0].imshow(in_img, vmin=vmin, vmax=vmax)
        axs[0].set_title("Input Full")
        axs[0].axis("off")

        im1 = axs[1].imshow(tgt_img, vmin=vmin, vmax=vmax)
        axs[1].set_title("Target Full (tmax_obs)")
        axs[1].axis("off")

        im2 = axs[2].imshow(prd_img, vmin=vmin, vmax=vmax)
        axs[2].set_title("Predicted Full (tiled)")
        axs[2].axis("off")

        fig.colorbar(im2, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
        plt.tight_layout()

        fname = os.path.join(
            self.out_dir,
            "epoch_png_full",
            f"epoch{epoch:03d}_full_tiled.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close(fig)

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0

        # Clear any cached memory from previous epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, (x, y) in enumerate(
            tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}", leave=False)
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Extra safety: remove NaN/Inf from input/target
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y)

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss, skip step & reduce LR")
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.5
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.detach().cpu())

        return running_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        psnr_list = []
        ssim_list = []
        ssim_metric = SSIMLoss().to(self.device)

        first_batch_saved = False

        for i, (x, y) in enumerate(
            tqdm(self.val_loader, desc=f"[Val] Epoch {epoch}", leave=False)
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            with amp.autocast(enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y)

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss in validation, skip batch")
                continue

            total_loss += float(loss.detach().cpu())
            # Patch-wise (normalized) PSNR/SSIM
            psnr_list.append(self._psnr(pred, y))
            ssim_val = 1.0 - float(ssim_metric(pred, y).detach().cpu())
            ssim_list.append(ssim_val)

            # Save patch PNGs for the first batch only (per epoch)
            if not first_batch_saved:
                self._save_epoch_png(epoch, x, y, pred, max_samples=1)
                first_batch_saved = True

        # Full-domain (200x280) inference + PSNR/SSIM PNG and logs
        self._save_full_image_tiled(epoch)

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        return avg_loss, avg_psnr, avg_ssim

    def fit(self):
        print(f"[TRAIN] Start training for {self.max_epochs} epochs on {self.device}")
        for epoch in range(1, self.max_epochs + 1):
            tr_loss = self.train_epoch(epoch)
            val_loss, val_psnr, val_ssim = self.validate(epoch)

            cur_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_PSNR={val_psnr:.2f}dB | val_SSIM={val_ssim:.4f} | "
                f"LR={cur_lr:.2e}"
            )

            # Save best model based on validation loss
            if val_loss < self.best_val - 1e-6:
                self.best_val = val_loss
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "stats": self.stats,   # for inverse normalization in inference
                }
                save_path = os.path.join(self.out_dir, "best_model.pth")
                torch.save(ckpt, save_path)
                print(f"  ✅ Save best model → {save_path}")


@torch.no_grad()
def inference_and_save_netcdf(
    model: nn.Module,
    x_all: np.ndarray,           # [T, H, W] input data (physical units)
    y_all: np.ndarray,           # [T, H, W] target data (physical units)
    stats: dict,                 # mean, std
    time_vals: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    device: str = "cuda:0",
    tile_size: int = 64,
    tile_overlap: int = 16,
    output_nc_path: str = "tmax_predicted.nc",
    batch_size: int = 8,         # process multiple time steps per batch
):
    """
    Perform tiled inference for all time steps and save results to NetCDF.
    """
    import xarray as xr

    model.eval()
    mean = float(stats["mean"])
    std = float(stats["std"])

    T, H, W = x_all.shape
    predictions = np.zeros_like(x_all, dtype=np.float32)  # [T, H, W]

    print(f"[INFERENCE] Processing {T} timesteps with tile size {tile_size}x{tile_size}")

    # Process in mini-batches
    for t_start in tqdm(range(0, T, batch_size), desc="Inference"):
        t_end = min(t_start + batch_size, T)
        batch_indices = range(t_start, t_end)

        for t in batch_indices:
            # Prepare input
            x_slice = x_all[t]  # [H, W]
            x_tensor = torch.from_numpy(x_slice).float().unsqueeze(0)  # [1, H, W]

            # Normalize
            x_norm = (x_tensor - mean) / std

            # Tiled inference
            pred_norm = predict_full_image_tiled(
                model,
                x_norm,
                tile_size=tile_size,
                overlap=tile_overlap,
                device=device,
            )  # [1, H, W]

            # Inverse normalization
            pred_phys = (pred_norm * std + mean).cpu().numpy()[0]  # [H, W]
            predictions[t] = pred_phys

    # Save NetCDF
    print(f"[INFERENCE] Saving predictions to {output_nc_path}")

    ds = xr.Dataset(
        {
            "tmax_predicted": (["time", "lat", "lon"], predictions),
            "tmax_input": (["time", "lat", "lon"], x_all),
            "tmax_target": (["time", "lat", "lon"], y_all),
        },
        coords={
            "time": time_vals,
            "lat": lat_vals,
            "lon": lon_vals,
        },
    )

    # Metadata
    ds["tmax_predicted"].attrs = {
        "long_name": "Predicted maximum temperature",
        "units": "K",
        "description": "Restormer model prediction"
    }
    ds["tmax_input"].attrs = {
        "long_name": "Input maximum temperature",
        "units": "K",
    }
    ds["tmax_target"].attrs = {
        "long_name": "Target maximum temperature (observed)",
        "units": "K",
    }

    ds.to_netcdf(output_nc_path)
    print(f"[INFERENCE] ✅ Saved to {output_nc_path}")

    # Compute statistics
    print("\n[INFERENCE] Computing metrics...")
    psnr_vals = []
    ssim_vals = []

    for t in tqdm(range(T), desc="Metrics"):
        pred = predictions[t]
        target = y_all[t]
        psnr_vals.append(psnr_np(pred, target))
        ssim_vals.append(ssim_np(pred, target))

    print(f"[INFERENCE] Mean PSNR: {np.mean(psnr_vals):.3f} dB")
    print(f"[INFERENCE] Mean SSIM: {np.mean(ssim_vals):.4f}")

    return predictions, ds


# ---------------------------------------------------------
# 6. main
# ---------------------------------------------------------
def main():
    CFG = SimpleNamespace(
        # --- Obs / target ---
        obs_nc_path="tas_all_methods_comparison.nc",
        target_var="tas_obs",

        # --- GCM input settings ---
        use_gcm_input=True,             # If True, use tasmin_raw as input
        gcm_nc_path="tas_all_methods_comparison.nc",
        gcm_var="tas_gcm",              # Replace with the actual variable name in tasmin_raw.nc

        # (Input variable when use_gcm_input=False)
        input_var="tas_sr",             # Input variable inside obs_nc_path

        device="cuda:0" if torch.cuda.is_available() else "cpu",

        # Training on 64x64 patches
        batch_size=32,
        val_batch_size=32,
        patch_size=64,
        val_full_image=False,  # Validation also uses 64x64; full images are handled separately via tiled inference

        val_ratio=0.2,
        seed=42,
        max_epochs=150,
        lr=1e-4,
        weight_decay=0.0,
        out_dir="checkpoints_restormer_tas_irgcm",
        use_amp=False,

        # Time index used for full-domain visualization (e.g., first time step)
        vis_time_index=0,
        tile_overlap=16,

        # Inference settings
        do_inference=True,                    # Whether to perform inference
        inference_checkpoint="checkpoints_restormer_tas_irgcm/best_model.pth",
        inference_output_nc="tas_restormer_prediction.nc",
    )

    set_seed(CFG.seed)
    print_device()

    # 1) Load NetCDF (depending on whether GCM input is used)
    if CFG.use_gcm_input:
        print("[DATA] Using GCM tasmax_raw as input (interpolated to obs grid).")
        x, y, time_vals, lat_vals, lon_vals = load_gcm_regridded_to_obs(
            obs_nc_path=CFG.obs_nc_path,
            target_var=CFG.target_var,
            gcm_nc_path=CFG.gcm_nc_path,
            gcm_var=CFG.gcm_var,
        )
    else:
        print("[DATA] Using variable from obs_nc_path as input.")
        x, y, time_vals, lat_vals, lon_vals = load_tmax_nc(
            CFG.obs_nc_path, CFG.input_var, CFG.target_var
        )

    print(f"[DATA] time={x.shape[0]}, lat={x.shape[1]}, lon={x.shape[2]}")

    # 2) DataLoaders
    train_loader, val_loader, stats = build_dataloaders(
        x,
        y,
        batch_size=CFG.batch_size,
        val_ratio=CFG.val_ratio,
        seed=CFG.seed,
        patch_size=CFG.patch_size,
        val_full_image=CFG.val_full_image,
        val_batch_size=CFG.val_batch_size,
    )

    # Example full-domain images (physical units numpy arrays)
    vis_t = int(np.clip(CFG.vis_time_index, 0, x.shape[0] - 1))
    full_x_example = x[vis_t]  # [H,W], input (GCM or tmax_vit)
    full_y_example = y[vis_t]  # [H,W], tmax_obs

    # 3) Model
    model = build_restormer(inp_channels=1, out_channels=1)

    # 4) Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        stats=stats,
        device=CFG.device,
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        max_epochs=CFG.max_epochs,
        out_dir=CFG.out_dir,
        use_amp=CFG.use_amp,
        full_x_example=full_x_example,
        full_y_example=full_y_example,
        tile_size=CFG.patch_size,
        tile_overlap=CFG.tile_overlap,
    )

    # trainer.fit()
    print("[DONE] Training finished.")

    # Load trained model
    if os.path.exists(CFG.inference_checkpoint):
        checkpoint = torch.load(CFG.inference_checkpoint, map_location=CFG.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFERENCE] Loaded checkpoint from {CFG.inference_checkpoint}")
        print(f"[INFERENCE] Best validation loss: {checkpoint['val_loss']:.6f}")
    else:
        print(f"[WARNING] Checkpoint not found: {CFG.inference_checkpoint}")
        print("[INFERENCE] Using current model state")

    # Perform inference
    predictions, ds = inference_and_save_netcdf(
        model=model,
        x_all=x,
        y_all=y,
        stats=stats,
        time_vals=time_vals,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        device=CFG.device,
        tile_size=CFG.patch_size,
        tile_overlap=CFG.tile_overlap,
        output_nc_path=CFG.inference_output_nc,
        batch_size=8,
    )

    print("[DONE] Inference completed and saved to NetCDF.")


if __name__ == "__main__":
    main()
