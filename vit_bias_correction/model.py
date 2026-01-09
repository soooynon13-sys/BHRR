# model.py
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class ViTBlock(nn.Module):
    """Standard Vision Transformer block."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """
    Simple ViT for bias correction with an optional fixed output range.

    Key points
    ----------
    - Optional Sigmoid on the output to enforce [0, 1] range.
    - Intended to be used together with a fixed physical range (e.g., 260–320 K).
    """
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=1,
        out_chans=1,
        embed_dim=256,
        depth=16,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.5,
        use_sigmoid=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_sigmoid = use_sigmoid

        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Simple decoder: project token embeddings back to pixel space
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, patch_size * patch_size * out_chans),
        )

        # Smoothing layer to reduce block artifacts
        self.smooth = nn.Sequential(
            nn.Conv2d(out_chans, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_chans, 3, padding=1),
        )

        # Output activation for range constraint
        if self.use_sigmoid:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)

        # Positional embedding (with interpolation if the grid size changed)
        B, N, E = x.shape  # N = number of patches from input
        N_pos = self.pos_embed.shape[1]  # number of patches at initialization

        if N != N_pos:
            # Interpolate positional embedding assuming a square grid at init
            N_pos_side = int(N_pos ** 0.5)
            if N_pos_side * N_pos_side != N_pos:
                raise ValueError(
                    "Positional embedding interpolation assumes a square "
                    "patch grid at initialization."
                )

            # New patch grid size
            h_patch = H // self.patch_size
            w_patch = W // self.patch_size

            # Reshape pos_embed to 2D
            pos_embed_2d = self.pos_embed.reshape(1, N_pos_side, N_pos_side, E)
            pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)  # (1, E, H, W)

            # Interpolate
            pos_embed_interp = nn.functional.interpolate(
                pos_embed_2d,
                size=(h_patch, w_patch),
                mode="bilinear",
                align_corners=False,
            )

            # Back to sequence
            pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).reshape(1, N, E)
            x = x + pos_embed_interp
        else:
            x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Decode per patch
        x = self.decoder(x)

        # Rearrange back to image
        h_patch = H // self.patch_size
        w_patch = W // self.patch_size

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_patch,
            w=w_patch,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # Interpolate back to original size if needed
        if x.shape[-2:] != (H, W):
            x = nn.functional.interpolate(
                x, size=(H, W), mode="bilinear", align_corners=False
            )

        # Final smoothing
        x = self.smooth(x)

        # Apply output activation (e.g. Sigmoid to constrain to [0, 1])
        x = self.output_activation(x)

        return x


def create_vit_fixed_range(
    img_size=256,
    patch_size=4,
    embed_dim=256,
    depth=16,
    num_heads=8,
    dropout=0.5,
):
    """
    Convenience wrapper: create a ViT with Sigmoid output
    intended for use with a fixed physical range.
    """
    return SimpleViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        out_chans=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=dropout,
        use_sigmoid=True,
    )


def create_vit_standard(
    img_size=256,
    patch_size=4,
    embed_dim=256,
    depth=16,
    num_heads=8,
    dropout=0.5,
):
    """
    Convenience wrapper: create a standard ViT
    with no output Sigmoid (unconstrained range).
    """
    return SimpleViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        out_chans=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=dropout,
        use_sigmoid=False,
    )


if __name__ == "__main__":
    # Simple self-test
    print("=" * 60)
    print("Testing SimpleViT with fixed output range")
    print("=" * 60)

    # Create model
    model_fixed = create_vit_fixed_range(
        img_size=240,
        patch_size=8,
        embed_dim=512,
        depth=12,
        num_heads=8,
        dropout=0.1,
    )

    print(f"\nModel created with Sigmoid: {model_fixed.use_sigmoid}")
    print(
        "  Parameters:",
        f"{sum(p.numel() for p in model_fixed.parameters()) / 1e6:.2f}M",
    )

    # Test input (normalized to [0, 1] range)
    x = torch.rand(2, 1, 240, 240)  # [B, C, H, W]
    print("\nInput:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{x.min():.3f}, {x.max():.3f}]")

    # Forward pass
    model_fixed.eval()
    with torch.no_grad():
        y = model_fixed(x)

    print("\nOutput:")
    print(f"  Shape: {y.shape}")
    print(f"  Range: [{y.min():.3f}, {y.max():.3f}]")

    # Check output range
    assert y.min() >= 0 and y.max() <= 1, "Output is not in [0, 1]."
    print("\nOutput is properly constrained to [0, 1].")

    print("\n" + "=" * 60)
