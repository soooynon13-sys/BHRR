# main.py
"""
Complete Bias Correction Pipeline (Fixed Range)

Workflow:
1. Data Preparation - Generate historical quantile maps
2. Model Training   - Train ViT
3. Inference        - Correct SSP scenarios
"""

import os
import argparse
import xarray as xr
from pathlib import Path

# Local modules
from utils import (
    prepare_all_quantile_maps_fixed_range,
    inspect_quantile_cache,
    print_gpu_memory,
    clear_gpu_memory
)
from train import train_fixed_range, get_default_config
from inference2 import run_bias_correction_fixed_range, run_multiple_scenarios
from model import SimpleViT


def setup_directories(base_dir='./'):
    """Create required directories."""
    dirs = {
        'cache': os.path.join(base_dir, 'cache'),
        'checkpoints': os.path.join(base_dir, 'checkpoints/vit_fixed_range'),
        'plots': os.path.join(base_dir, 'plots/vit_fixed_range'),
        'outputs': os.path.join(base_dir, 'outputs')
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"{name:12s}: {path}")

    return dirs


def step1_prepare_data(
    gcm_hist_path,
    obs_hist_path,
    dirs,
    n_quantiles=500,
    lr_range=(260, 320),
    hr_range=(260, 320),
    device='cuda:0',
    force_recompute=False
):
    """
    Step 1: Data preparation - generate historical quantile maps.

    Parameters
    ----------
    gcm_hist_path : str
        Path to historical GCM data.
    obs_hist_path : str
        Path to historical observational data.
    dirs : dict
        Directory information (cache, checkpoints, plots, outputs).
    n_quantiles : int
        Number of quantiles.
    lr_range, hr_range : tuple
        Fixed range (min, max) for LR and HR in Kelvin.
    device : str
        GPU device for quantile computation.
    force_recompute : bool
        If True, ignore existing cache and recompute.

    Returns
    -------
    cache_path : str
        Path to the generated cache file.
    """

    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)

    # Cache path
    cache_filename = f'quantile_maps_q{n_quantiles}_{lr_range[0]}_{lr_range[1]}.pkl'
    cache_path = os.path.join(dirs['cache'], cache_filename)

    # Use cache if it exists
    if os.path.exists(cache_path) and not force_recompute:
        print(f"\nCache exists: {cache_path}")
        inspect_quantile_cache(cache_path)
        return cache_path

    # Load data
    print("\nLoading data...")
    print(f"  GCM historical: {gcm_hist_path}")
    print(f"  OBS historical: {obs_hist_path}")

    gcm_hist = xr.open_dataset(gcm_hist_path)
    obs_hist = xr.open_dataset(obs_hist_path)

    # Detect variable names (3D variables)
    gcm_var = [v for v in gcm_hist.data_vars if gcm_hist[v].ndim == 3][0]
    obs_var = [v for v in obs_hist.data_vars if obs_hist[v].ndim == 3][0]

    print(f"  GCM variable: {gcm_var}")
    print(f"  OBS variable: {obs_var}")

    # Train/val split (80/20)
    print("\nSplitting train/val (80/20)...")

    gcm_times = gcm_hist.time.values
    obs_times = obs_hist.time.values

    split_idx_gcm = int(len(gcm_times) * 0.8)
    split_idx_obs = int(len(obs_times) * 0.8)

    lr_train = gcm_hist[gcm_var].isel(time=slice(0, split_idx_gcm))
    lr_val = gcm_hist[gcm_var].isel(time=slice(split_idx_gcm, None))
    hr_train = obs_hist[obs_var].isel(time=slice(0, split_idx_obs))
    hr_val = obs_hist[obs_var].isel(time=slice(split_idx_obs, None))

    print(f"  LR train: {lr_train.shape} ({lr_train.time.values[0]} ~ {lr_train.time.values[-1]})")
    print(f"  LR val:   {lr_val.shape} ({lr_val.time.values[0]} ~ {lr_val.time.values[-1]})")
    print(f"  HR train: {hr_train.shape} ({hr_train.time.values[0]} ~ {hr_train.time.values[-1]})")
    print(f"  HR val:   {hr_val.shape} ({hr_val.time.values[0]} ~ {hr_val.time.values[-1]})")

    # Compute quantile maps
    print("\nComputing quantile maps...")
    result = prepare_all_quantile_maps_fixed_range(
        lr_train, hr_train,
        lr_val, hr_val,
        n_quantiles=n_quantiles,
        lr_range=lr_range,
        hr_range=hr_range,
        cache_path=cache_path,
        force_recompute=force_recompute,
        device=device,
        batch_size=2000
    )

    print("\nData preparation complete.")
    print(f"  Cache saved: {cache_path}")

    # Inspect cache
    inspect_quantile_cache(cache_path)

    return cache_path


def step2_train_model(
    cache_path,
    dirs,
    config=None,
    device='cuda:1',
    resume_from=None
):
    """
    Step 2: Train the ViT model.

    Parameters
    ----------
    cache_path : str
        Path to the quantile-map cache file.
    dirs : dict
        Directory information.
    config : dict or None
        Training configuration (if None, use default config).
    device : str
        GPU device for training.
    resume_from : str or None
        Path to checkpoint to resume from (not implemented yet).

    Returns
    -------
    checkpoint_path : str
        Path to the trained model checkpoint.
    """

    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)

    # Load quantile maps
    print(f"\nLoading quantile maps from: {cache_path}")
    import pickle
    with open(cache_path, 'rb') as f:
        qdata = pickle.load(f)

    lr_q_train, hr_q_train = qdata['train']
    lr_q_val, hr_q_val = qdata['val']
    quantiles = qdata['quantiles']
    normalization = qdata['normalization']

    print(f"  Train: LR={lr_q_train.shape}, HR={hr_q_train.shape}")
    print(f"  Val:   LR={lr_q_val.shape}, HR={hr_q_val.shape}")
    print(f"  Normalization: {normalization['type']}")

    # Configuration
    if config is None:
        config = get_default_config(use_fixed_range=True)

    # Set directories and device in config
    config['checkpoint_dir'] = dirs['checkpoints']
    config['plot_dir'] = dirs['plots']
    config['device'] = device

    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    # Check GPU memory
    print_gpu_memory(int(device.split(':')[1]) if ':' in device else None)

    # Train
    print("\nStarting training...")

    if resume_from:
        print(f"  Resuming from: {resume_from}")
        # TODO: implement resume logic if needed

    trainer = train_fixed_range(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        lr_range=(normalization['lr_min'], normalization['lr_max']),
        hr_range=(normalization['hr_min'], normalization['hr_max']),
        custom_config=config
    )

    checkpoint_path = os.path.join(dirs['checkpoints'], 'best_model.pth')

    print("\nTraining complete.")
    print(f"  Best loss: {trainer.best_loss:.6f}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Clear GPU memory
    clear_gpu_memory(int(device.split(':')[1]) if ':' in device else None)

    return checkpoint_path


def step3_inference(
    ssp_paths,
    checkpoint_path,
    cache_path,
    dirs,
    device='cuda:1',
    batch_size=32,
    var_name='tasmax'
):
    """
    Step 3: SSP scenario inference and correction.

    Parameters
    ----------
    ssp_paths : dict
        Scenario file paths, e.g. {'ssp245': '/path/to/ssp245.nc', ...}.
    checkpoint_path : str
        Path to the trained model checkpoint.
    cache_path : str
        Path to the quantile-map cache file.
    dirs : dict
        Directory information.
    device : str
        GPU device.
    batch_size : int
        Batch size for model inference.
    var_name : str
        Variable name in the SSP NetCDF files.

    Returns
    -------
    results : dict
        Dictionary mapping scenario name to corrected xarray.DataArray.
    """

    print("\n" + "=" * 80)
    print("STEP 3: INFERENCE (SSP SCENARIOS)")
    print("=" * 80)

    # Model configuration
    print("\nModel configuration:")
    model_config = {
        'img_size': 240,
        'patch_size': 8,
        'in_chans': 1,
        'out_chans': 1,
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'dropout': 0.1,
        'use_sigmoid': True  # fixed-range output
    }

    for key, value in model_config.items():
        print(f"  {key:20s}: {value}")

    # Check GPU memory
    print_gpu_memory(int(device.split(':')[1]) if ':' in device else None)

    # Process each scenario
    results = {}

    for i, (scenario, input_path) in enumerate(ssp_paths.items(), 1):
        print("\n" + "=" * 80)
        print(f"Processing {i}/{len(ssp_paths)}: {scenario}")
        print("=" * 80)

        if not os.path.exists(input_path):
            print("  Skipping: input file not found.")
            continue

        output_path = os.path.join(dirs['outputs'], f'vit_{scenario}_corrected.nc')

        try:
            corrected = run_bias_correction_fixed_range(
                lr_data_path=input_path,
                checkpoint_path=checkpoint_path,
                quantile_map_path=cache_path,
                output_path=output_path,
                model_class=SimpleViT,
                model_config=model_config,
                device=device,
                batch_size=batch_size,
                var_name=var_name
            )

            results[scenario] = corrected
            print(f"  {scenario} completed.")
        except Exception as e:
            print(f"  {scenario} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nInference complete.")
    print(f"  Processed: {len(results)}/{len(ssp_paths)} scenarios")

    # Clear GPU memory
    clear_gpu_memory(int(device.split(':')[1]) if ':' in device else None)

    return results


def run_full_pipeline(
    gcm_hist_path,
    obs_hist_path,
    ssp_paths,
    base_dir='./',
    n_quantiles=500,
    lr_range=(260, 320),
    hr_range=(260, 320),
    num_epochs=200,
    data_device='cuda:0',
    train_device='cuda:1',
    inference_device='cuda:1',
    skip_data_prep=False,
    skip_training=False,
    force_recompute=False
):
    """
    Run the full fixed-range bias-correction pipeline.

    Parameters
    ----------
    gcm_hist_path : str
        Historical GCM data path.
    obs_hist_path : str
        Historical observation data path.
    ssp_paths : dict
        Scenario paths, e.g. {'ssp245': '/path/to/ssp245.nc', ...}.
    base_dir : str
        Base working directory.
    n_quantiles : int
        Number of quantiles.
    lr_range, hr_range : tuple
        Fixed range for LR and HR (min, max).
    num_epochs : int
        Number of training epochs.
    data_device : str
        GPU device for data preparation.
    train_device : str
        GPU device for training.
    inference_device : str
        GPU device for inference.
    skip_data_prep : bool
        If True, skip data preparation and use existing cache.
    skip_training : bool
        If True, skip training and use existing checkpoint.
    force_recompute : bool
        If True, recompute quantile maps even if cache exists.

    Returns
    -------
    results : dict
        Summary dictionary with cache path, checkpoint path, and scenario results.
    """

    print("\n" + "=" * 80)
    print("FULL BIAS CORRECTION PIPELINE")
    print("=" * 80)
    print("\nInput files:")
    print(f"  GCM historical: {gcm_hist_path}")
    print(f"  OBS historical: {obs_hist_path}")
    print("  SSP scenarios:")
    for scenario, path in ssp_paths.items():
        print(f"    {scenario:10s}: {path}")
    print("\nConfiguration:")
    print(f"  Base directory: {base_dir}")
    print(f"  N quantiles:    {n_quantiles}")
    print(f"  LR range:       {lr_range} K")
    print(f"  HR range:       {hr_range} K")
    print(f"  Num epochs:     {num_epochs}")
    print(f"  Data device:    {data_device}")
    print(f"  Train device:   {train_device}")
    print(f"  Infer device:   {inference_device}")

    # Directories
    print("\nSetting up directories...")
    dirs = setup_directories(base_dir)

    # Step 1: Data preparation
    if not skip_data_prep:
        cache_path = step1_prepare_data(
            gcm_hist_path=gcm_hist_path,
            obs_hist_path=obs_hist_path,
            dirs=dirs,
            n_quantiles=n_quantiles,
            lr_range=lr_range,
            hr_range=hr_range,
            device=data_device,
            force_recompute=force_recompute
        )
    else:
        print("\nSkipping data preparation.")
        cache_filename = f'quantile_maps_q{n_quantiles}_{lr_range[0]}_{lr_range[1]}.pkl'
        cache_path = os.path.join(dirs['cache'], cache_filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        print(f"  Using existing cache: {cache_path}")

    # Step 2: Training
    if not skip_training:
        config = get_default_config(use_fixed_range=True)
        config['num_epochs'] = num_epochs

        checkpoint_path = step2_train_model(
            cache_path=cache_path,
            dirs=dirs,
            config=config,
            device=train_device
        )
    else:
        print("\nSkipping training.")
        checkpoint_path = os.path.join(dirs['checkpoints'], 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"  Using existing checkpoint: {checkpoint_path}")

    # Step 3: Inference
    results = step3_inference(
        ssp_paths=ssp_paths,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        dirs=dirs,
        device=inference_device,
        batch_size=32,
        var_name='tasmax'
    )

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Data prepared:       {cache_path}")
    print(f"  Model checkpoint:    {checkpoint_path}")
    print(f"  Scenarios processed: {len(results)}/{len(ssp_paths)}")

    if results:
        print("\nOutput files:")
        for scenario in results.keys():
            output_path = os.path.join(dirs['outputs'], f'vit_{scenario}_corrected.nc')
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / 1024 ** 2
                print(f"  {scenario:10s}: {output_path} ({size_mb:.2f} MB)")

    print("\n" + "=" * 80 + "\n")

    return {
        'cache_path': cache_path,
        'checkpoint_path': checkpoint_path,
        'results': results,
        'dirs': dirs
    }
