# train.py
import torch
from torch.utils.data import DataLoader
from model import SimpleViT
from engine import BiasCorrectTrainer, QuantileDataset


def train_bias_correction(
    lr_q_train,
    hr_q_train,
    lr_q_val,
    hr_q_val,
    quantiles,
    config,
    normalization=None,
    lr_stats=None,  # deprecated (kept for backward compatibility)
    hr_stats=None,  # deprecated (kept for backward compatibility)
):
    """
    Train a bias-correction model using quantile maps.

    Parameters
    ----------
    lr_q_train, hr_q_train : numpy.ndarray
        Training quantile maps, shape [n_quantiles, H, W], normalized to [0, 1].
    lr_q_val, hr_q_val : numpy.ndarray
        Validation quantile maps, shape [n_quantiles, H, W], normalized to [0, 1].
    quantiles : numpy.ndarray
        Quantile levels in percent [0, 100].
    config : dict
        Training configuration.
    normalization : dict or None
        Normalization information. Example:
        {'type': 'fixed_range', 'lr_min': 260, 'lr_max': 320, 'hr_min': 260, 'hr_max': 320}
    lr_stats, hr_stats : dict or None
        Deprecated Z-score statistics (for backward compatibility).
    """

    print("=" * 60)
    print("Starting Training Pipeline")
    print("=" * 60)

    # 1. Check data ranges
    print("\nInput data info:")
    print(
        f"  LR train: {lr_q_train.shape}, "
        f"range=[{lr_q_train.min():.3f}, {lr_q_train.max():.3f}]"
    )
    print(
        f"  HR train: {hr_q_train.shape}, "
        f"range=[{hr_q_train.min():.3f}, {hr_q_train.max():.3f}]"
    )
    print(
        f"  LR val:   {lr_q_val.shape}, "
        f"range=[{lr_q_val.min():.3f}, {lr_q_val.max():.3f}]"
    )
    print(
        f"  HR val:   {hr_q_val.shape}, "
        f"range=[{hr_q_val.min():.3f}, {hr_q_val.max():.3f}]"
    )

    if normalization is not None:
        print("\nNormalization:")
        print(f"  Type: {normalization['type']}")
        if normalization["type"] == "fixed_range":
            print(
                f"  LR: [{normalization['lr_min']} K, {normalization['lr_max']} K]"
            )
            print(
                f"  HR: [{normalization['hr_min']} K, {normalization['hr_max']} K]"
            )

    # 2. Create datasets (with range verification)
    print("\nCreating datasets...")
    train_dataset = QuantileDataset(
        lr_q_train,
        hr_q_train,
        verify_range=True,
    )
    val_dataset = QuantileDataset(
        lr_q_val,
        hr_q_val,
        verify_range=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # 3. Create model
    print("\nCreating model...")

    # Decide whether to use Sigmoid on the output
    use_sigmoid = normalization is not None and normalization.get("type") == "fixed_range"

    model = SimpleViT(
        img_size=lr_q_train.shape[1],
        patch_size=config["patch_size"],
        in_chans=1,
        out_chans=1,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=4.0,
        dropout=config["dropout"],
        use_sigmoid=use_sigmoid,
    )

    print(f"  Model: {type(model).__name__}")
    print(f"  Image size: {lr_q_train.shape[1]}x{lr_q_train.shape[2]}")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim:  {config['embed_dim']}")
    print(f"  Depth:      {config['depth']}")
    print(f"  Num heads:  {config['num_heads']}")
    print(f"  Dropout:    {config['dropout']}")
    print(f"  Use Sigmoid: {use_sigmoid}")
    print(
        "  Parameters:",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
    )

    # 4. Create trainer
    print("\nCreating trainer...")
    trainer = BiasCorrectTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        checkpoint_dir=config["checkpoint_dir"],
        plot_dir=config["plot_dir"],
        plot_every=config["plot_every"],
        quantiles=quantiles,
        normalization=normalization,
        # backward compatibility
        lr_stats=lr_stats,
        hr_stats=hr_stats,
    )

    # 5. Set learning-rate scheduler
    trainer.set_scheduler("cosine", config["num_epochs"])

    # 6. Train
    print("\n" + "=" * 60)
    trainer.fit(config["num_epochs"], lr_q_val)

    return trainer


def get_default_config(use_fixed_range=True):
    """
    Default training configuration.

    Parameters
    ----------
    use_fixed_range : bool
        If True, use the fixed-range method (with Sigmoid and fixed physical range).
        If False, use the legacy Z-score method.
    """
    if use_fixed_range:
        # Fixed-range method (more stable in many cases)
        return {
            "batch_size": 8,
            "num_workers": 4,
            # SimpleViT with Sigmoid
            "patch_size": 8,
            "embed_dim": 512,
            "depth": 12,
            "num_heads": 8,
            "dropout": 0.1,
            "num_epochs": 200,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "device": "cuda:1",
            "checkpoint_dir": "checkpoints/vit_fixed_range",
            "plot_dir": "plots/vit_fixed_range",
            "plot_every": 5,
        }
    else:
        # Legacy Z-score method
        return {
            "batch_size": 4,
            "num_workers": 4,
            # SimpleViT without Sigmoid
            "patch_size": 4,
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "num_epochs": 200,
            "lr": 5e-5,
            "weight_decay": 0.01,
            "device": "cuda:1",
            "checkpoint_dir": "checkpoints/simple_vit",
            "plot_dir": "plots/simple_vit",
            "plot_every": 5,
        }


def train_fixed_range(
    lr_q_train,
    hr_q_train,
    lr_q_val,
    hr_q_val,
    quantiles,
    lr_range=(260, 320),
    hr_range=(260, 320),
    custom_config=None,
):
    """
    Convenience function for training with the fixed-range method.

    Parameters
    ----------
    lr_q_train, hr_q_train : numpy.ndarray
        Training quantile maps [n_quantiles, H, W], normalized to [0, 1].
    lr_q_val, hr_q_val : numpy.ndarray
        Validation quantile maps [n_quantiles, H, W], normalized to [0, 1].
    quantiles : numpy.ndarray
        Quantile levels in percent.
    lr_range, hr_range : tuple
        Physical ranges for LR and HR in Kelvin, e.g. (260, 320).
    custom_config : dict or None
        Custom training configuration. If None, use default config.
    """

    normalization = {
        "type": "fixed_range",
        "lr_min": lr_range[0],
        "lr_max": lr_range[1],
        "hr_min": hr_range[0],
        "hr_max": hr_range[1],
    }

    if custom_config is None:
        config = get_default_config(use_fixed_range=True)
    else:
        config = custom_config

    trainer = train_bias_correction(
        lr_q_train,
        hr_q_train,
        lr_q_val,
        hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=normalization,
    )

    return trainer


def train_standard(
    lr_q_train,
    hr_q_train,
    lr_q_val,
    hr_q_val,
    lr_stats,
    hr_stats,
    quantiles,
    custom_config=None,
):
    """
    Convenience function for legacy Z-score training.

    This is kept for backward compatibility.
    """

    if custom_config is None:
        config = get_default_config(use_fixed_range=False)
    else:
        config = custom_config

    trainer = train_bias_correction(
        lr_q_train,
        hr_q_train,
        lr_q_val,
        hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=None,
        lr_stats=lr_stats,
        hr_stats=hr_stats,
    )

    return trainer


if __name__ == "__main__":
    # Example usage
    import pickle

    print("=" * 60)
    print("Training Script - Usage Example")
    print("=" * 60)

    # 1. Load data
    print("\nLoading data...")
    with open("quantile_maps_fixed_range.pkl", "rb") as f:
        qdata = pickle.load(f)

    lr_q_train, hr_q_train = qdata["train"]
    lr_q_val, hr_q_val = qdata["val"]
    quantiles = qdata["quantiles"]
    normalization = qdata["normalization"]

    print(f"  Loaded: {lr_q_train.shape}")

    # 2. Method 1: fixed-range training
    print("\nMethod 1: Fixed Range Training")

    trainer = train_fixed_range(
        lr_q_train,
        hr_q_train,
        lr_q_val,
        hr_q_val,
        quantiles=quantiles,
        lr_range=(normalization["lr_min"], normalization["lr_max"]),
        hr_range=(normalization["hr_min"], normalization["hr_max"]),
        custom_config=None,
    )

    print("\nTraining completed.")
    print(f"  Best loss: {trainer.best_loss:.6f}")

    # 3. Method 2: direct call with a config
    print("\nMethod 2: Direct Call")

    config = get_default_config(use_fixed_range=True)
    config["num_epochs"] = 100  # example: override epochs

    trainer2 = train_bias_correction(
        lr_q_train,
        hr_q_train,
        lr_q_val,
        hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=normalization,
    )

    print("\n" + "=" * 60)
