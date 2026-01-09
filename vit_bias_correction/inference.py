# inference.py
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy import interpolate
import os


def load_trained_model(checkpoint_path, model_class, device='cuda:0'):
    """Load a trained model checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    return checkpoint


def prepare_inference_quantile_maps_gpu(
    lr_data,
    n_quantiles,
    lr_mean,
    lr_std,
    device='cuda:0',
    batch_size=2000
):
    """
    Compute quantile maps for inference using GPU acceleration.

    Parameters
    ----------
    lr_data : xarray.DataArray
        LR data to be corrected, shape [T, lat, lon].
    n_quantiles : int
        Number of quantiles.
    lr_mean, lr_std : float
        Normalization statistics used during training (LR).
    device : str
        GPU device.
    batch_size : int
        Batch size for quantile computation.

    Returns
    -------
    lr_q_normalized : numpy.ndarray
        Normalized LR quantile maps, shape [n_quantiles, lat, lon].
    quantiles : numpy.ndarray
        Quantile levels in percent [0, 100].
    """
    print("\nComputing quantile maps for inference (GPU)...")
    print(f"  Data shape: {lr_data.shape}")
    print(f"  Device: {device}")

    T, H, W = lr_data.shape
    quantiles = np.linspace(0, 100, n_quantiles)

    # Quantile levels on GPU in [0, 1]
    q_levels = torch.linspace(0, 1, n_quantiles, device=device)

    # Output array
    lr_q_maps = np.zeros((n_quantiles, H, W), dtype=np.float32)

    # Reshape to [T, H*W]
    data_2d = lr_data.values.reshape(T, H * W)
    n_pixels = H * W

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_pixels, batch_size), desc="Computing quantiles (GPU)"):
            end_idx = min(start_idx + batch_size, n_pixels)

            batch_data = data_2d[:, start_idx:end_idx]

            # Process each pixel within the batch
            for local_idx in range(batch_data.shape[1]):
                pixel_data = batch_data[:, local_idx]
                valid = ~np.isnan(pixel_data)

                if valid.sum() >= 10:
                    # Move to GPU and compute quantiles
                    pixel_tensor = torch.FloatTensor(pixel_data[valid]).to(device)
                    q_vals = torch.quantile(pixel_tensor, q_levels)

                    # Store results
                    global_idx = start_idx + local_idx
                    lat_idx = global_idx // W
                    lon_idx = global_idx % W

                    lr_q_maps[:, lat_idx, lon_idx] = q_vals.cpu().numpy()
                else:
                    # Not enough valid data
                    global_idx = start_idx + local_idx
                    lat_idx = global_idx // W
                    lon_idx = global_idx % W
                    lr_q_maps[:, lat_idx, lon_idx] = np.nan

    # Normalize (same as in training)
    lr_q_normalized = (lr_q_maps - lr_mean) / lr_std
    lr_q_normalized = np.nan_to_num(lr_q_normalized, 0)

    print(f"Quantile maps ready: {lr_q_normalized.shape}")
    return lr_q_normalized, quantiles


def predict_quantile_maps(model, lr_q_normalized, device, batch_size=16):
    """
    Predict HR quantile maps using the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    lr_q_normalized : numpy.ndarray
        Normalized LR quantile maps, shape [n_quantiles, lat, lon].
    device : str
        Device for inference.
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    hr_q_predicted : numpy.ndarray
        Predicted HR quantile maps, same shape as lr_q_normalized.
    """
    print("\nPredicting quantile maps with the model...")

    model.eval()
    hr_q_predicted = []

    n_quantiles = len(lr_q_normalized)

    with torch.no_grad():
        for i in tqdm(range(0, n_quantiles, batch_size), desc="Predicting"):
            batch = lr_q_normalized[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)

            with torch.cuda.amp.autocast():
                pred = model(batch_tensor)

            hr_q_predicted.append(pred.cpu().squeeze(1).numpy())

    hr_q_predicted = np.concatenate(hr_q_predicted, axis=0)

    print(f"Prediction done: {hr_q_predicted.shape}")
    return hr_q_predicted


def apply_quantile_correction_gpu(
    lr_data,
    hr_q_predicted,
    quantiles,
    hr_mean,
    hr_std,
    device='cuda:0',
    batch_size=1000
):
    """
    Apply quantile-based bias correction using GPU-accelerated quantile computation.

    Parameters
    ----------
    lr_data : xarray.DataArray
        Original LR data [T, lat, lon].
    hr_q_predicted : numpy.ndarray
        Predicted HR quantile maps (normalized), shape [n_quantiles, lat, lon].
    quantiles : numpy.ndarray
        Quantile levels in percent [0, 100].
    hr_mean, hr_std : float
        HR normalization statistics (used for de-normalization).
    device : str
        GPU device.
    batch_size : int
        Batch size for processing pixels.

    Returns
    -------
    lr_corrected : xarray.DataArray
        Bias-corrected LR data.
    """
    print("\nApplying bias correction (GPU)...")

    # De-normalize HR quantile maps
    hr_q_maps = hr_q_predicted * hr_std + hr_mean

    # Initialize corrected data
    lr_corrected = lr_data.copy(deep=True)

    T, H, W = lr_data.shape
    n_pixels = H * W

    # Reshape to [T, H*W]
    data_2d = lr_data.values.reshape(T, H * W)
    corrected_2d = np.zeros_like(data_2d)

    # Quantile levels as tensor in [0, 1]
    q_levels = torch.FloatTensor(quantiles / 100.0).to(device)

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_pixels, batch_size), desc="Correcting (GPU)"):
            end_idx = min(start_idx + batch_size, n_pixels)

            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                lat_idx = global_idx // W
                lon_idx = global_idx % W

                lr_vals = data_2d[:, global_idx]
                hr_q = hr_q_maps[:, lat_idx, lon_idx]

                valid = ~np.isnan(lr_vals)
                if valid.sum() < 10:
                    continue

                # LR quantiles (GPU)
                lr_vals_gpu = torch.FloatTensor(lr_vals[valid]).to(device)
                lr_q_gpu = torch.quantile(lr_vals_gpu, q_levels)

                # HR quantiles (GPU)
                hr_q_gpu = torch.FloatTensor(hr_q).to(device)

                # Move to CPU for interpolation
                lr_q_cpu = lr_q_gpu.cpu().numpy()
                hr_q_cpu = hr_q_gpu.cpu().numpy()

                # Build transfer function
                transfer = interpolate.interp1d(
                    lr_q_cpu, hr_q_cpu,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(hr_q_cpu[0], hr_q_cpu[-1])
                )

                # Apply correction
                corrected_2d[:, global_idx] = transfer(lr_vals)

    # Reshape back to [T, H, W]
    lr_corrected.values = corrected_2d.reshape(T, H, W)

    print("Bias correction complete.")
    return lr_corrected


def run_bias_correction(
    lr_data_path,
    checkpoint_path,
    output_path,
    model_class,
    model_config,
    n_quantiles=100,
    device='cuda:1',
    batch_size=16,
    restore_extremes=False,
    use_gpu_quantiles=True,
    quantile_batch_size=2000,
    var_name=None,
    quantile_map_path=None
):
    """
    Full bias-correction pipeline with optional GPU acceleration.

    Parameters
    ----------
    lr_data_path : str
        Path to the input LR data (.nc).
    checkpoint_path : str
        Path to the model checkpoint.
    output_path : str
        Path for the corrected output file (.nc).
    model_class : class
        Model class.
    model_config : dict
        Model configuration dictionary.
    n_quantiles : int
        Number of quantiles.
    device : str
        Device for inference.
    batch_size : int
        Batch size for model inference.
    restore_extremes : bool
        Whether to apply an additional extreme-restoration step.
    use_gpu_quantiles : bool
        Whether to compute quantile maps using GPU.
    quantile_batch_size : int
        Batch size for quantile computation.
    var_name : str or None
        Variable name in the NetCDF file (auto-detected if None).
    quantile_map_path : str or None
        Optional path to a precomputed quantile-map pickle file.
    """

    print("=" * 60)
    print("Starting Bias Correction Pipeline (GPU Accelerated)")
    print("=" * 60)

    # 1. Load checkpoint
    checkpoint = load_trained_model(checkpoint_path, model_class, device)

    lr_mean, lr_std = checkpoint['lr_stats']
    hr_mean, hr_std = checkpoint['hr_stats']

    print("\nNormalization statistics:")
    print(f"  LR: mean={lr_mean:.4f}, std={lr_std:.4f}")
    print(f"  HR: mean={hr_mean:.4f}, std={hr_std:.4f}")

    # 2. Initialize model
    print("\nInitializing model...")
    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 3. Load LR data
    print(f"\nLoading LR data from: {lr_data_path}")
    lr_ds = xr.open_dataset(lr_data_path)

    # Auto-detect variable name if not given
    if var_name is None:
        var_candidates = [var for var in lr_ds.data_vars if lr_ds[var].ndim == 3]
        if not var_candidates:
            raise ValueError("No suitable 3D variable found in the dataset.")
        var_name = var_candidates[0]
        print(f"  Detected variable: {var_name}")
    lr_data = lr_ds[var_name]

    print(f"  Variable: {var_name}")
    print(f"  Shape: {lr_data.shape}")
    print(f"  Time range: {lr_data.time.values[0]} ~ {lr_data.time.values[-1]}")

    # 4. Quantile maps (either precomputed or computed on the fly)
    if quantile_map_path is not None and os.path.exists(quantile_map_path):
        print(f"  Loading precomputed quantile maps from: {quantile_map_path}")
        import pickle
        with open(quantile_map_path, 'rb') as f:
            quantile_data = pickle.load(f)
        lr_q_normalized = quantile_data['train'][0]
        quantiles = quantile_data['quantiles']
    else:
        print("  No precomputed quantile maps found.")
        print("  Computing quantile maps...")
        if use_gpu_quantiles:
            lr_q_normalized, quantiles = prepare_inference_quantile_maps_gpu(
                lr_data, n_quantiles, lr_mean, lr_std,
                device=device, batch_size=quantile_batch_size
            )
        else:
            # CPU version should be defined elsewhere
            from inference import prepare_inference_quantile_maps
            lr_q_normalized, quantiles = prepare_inference_quantile_maps(
                lr_data, n_quantiles, lr_mean, lr_std
            )

    # 5. Model prediction
    hr_q_predicted = predict_quantile_maps(
        model, lr_q_normalized, device, batch_size
    )

    # 6. Apply bias correction
    if use_gpu_quantiles:
        lr_corrected = apply_quantile_correction_gpu(
            lr_data, hr_q_predicted, quantiles, hr_mean, hr_std,
            device=device, batch_size=quantile_batch_size
        )
    else:
        # CPU version should be defined elsewhere
        from inference import apply_quantile_correction
        lr_corrected = apply_quantile_correction(
            lr_data, hr_q_predicted, quantiles, hr_mean, hr_std
        )

    # 7. Optional extreme restoration
    if restore_extremes:
        from inference import restore_extreme_values
        target_stats = {
            'mean': hr_mean,
            'std': hr_std,
            'q01': hr_mean - 3 * hr_std,
            'q99': hr_mean + 3 * hr_std
        }
        lr_corrected = restore_extreme_values(lr_corrected, target_stats)

    # 8. Save results
    print(f"\nSaving corrected data to: {output_path}")

    lr_corrected.attrs['bias_correction'] = 'Applied'
    lr_corrected.attrs['model'] = model_class.__name__
    lr_corrected.attrs['checkpoint'] = checkpoint_path
    lr_corrected.attrs['n_quantiles'] = n_quantiles

    output_ds = lr_corrected.to_dataset(name=var_name + '_corrected')
    output_ds.to_netcdf(output_path)

    print("Saved corrected dataset successfully.")

    # 9. Statistics comparison
    print("\nStatistics comparison:")
    print(f"{'Metric':<15} {'Original':<15} {'Corrected':<15}")
    print("-" * 45)
    print(f"{'Mean':<15} {float(lr_data.mean()):<15.4f} {float(lr_corrected.mean()):<15.4f}")
    print(f"{'Std':<15} {float(lr_data.std()):<15.4f} {float(lr_corrected.std()):<15.4f}")
    print(f"{'Min':<15} {float(lr_data.min()):<15.4f} {float(lr_corrected.min()):<15.4f}")
    print(f"{'Max':<15} {float(lr_data.max()):<15.4f} {float(lr_corrected.max()):<15.4f}")

    print("\nBias correction finished.")
    print("=" * 60)

    return lr_corrected


if __name__ == "__main__":
    # Example test run
    from model import SimpleViT

    model_config = {
        'img_size': 240,
        'patch_size': 8,
        'in_chans': 1,
        'out_chans': 1,
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'dropout': 0.1
    }

    lr_corrected = run_bias_correction(
        lr_data_path='./data/lr_monthly_max.nc',
        checkpoint_path='checkpoints/simple_vit/best_model.pth',
        output_path='./data/lr_corrected_monthly_max.nc',
        model_class=SimpleViT,
        model_config=model_config,
        n_quantiles=500,
        device='cuda:1',
        batch_size=32,
        use_gpu_quantiles=True,
        quantile_batch_size=2000,
        restore_extremes=False
    )
