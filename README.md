# BHRR: A Transformer Framework for Bias-Corrected High-Resolution Temperature Fields

This repository provides the official implementation of the **Bias-corrected High-Resolution Restoration (BHRR)** framework for daily near-surface air temperature fields. BHRR sequentially combines

1. **Restormer-based spatial restoration** of coarse-resolution CMIP6 GCM outputs, and  
2. **Vision Transformer (ViT)-based quantile-mapping bias correction**,

to generate **bias-corrected, high-resolution daily temperature fields** over a fixed regional domain.

The framework is demonstrated over the **Oceania** domain (200 × 280 grid) using ACCESS-CM2 CMIP6 simulations and the **Princeton Global Forcing v3 (PGFv3)** dataset as the observation-based reference. NEX-GDDP-CMIP6 is used as a benchmark downscaled product.

---

## 1. Related paper

This code accompanies the manuscript:

> **Song, Y.H., Kim, H.J., & Chung, E.-S.**  
> *A Transformer Framework for High-Resolution Temperature Fields with Integrated Bias Correction* (submitted).

Please cite this work if you use the BHRR framework or this repository in your research.

---

## 2. Framework overview

BHRR consists of two main stages:

1. **Restormer-based high-resolution restoration (`restormer_stage/`)**

   - Input: daily CMIP6 GCM near-surface air temperature (Tas, Tmin, Tmax) interpolated to the 200 × 280 target grid.  
   - Target: PGFv3 temperature fields on the same grid.  
   - The Restormer network learns to restore fine-scale spatial structure (coastlines, orography, island details) and improves PSNR/SSIM relative to raw linearly interpolated GCM fields.:contentReference[oaicite:1]{index=1}  

2. **ViT-based quantile bias correction (`vit_bias_correction/`)**

   - Inputs:  
     - Historical **low-resolution (LR)** and **high-resolution (HR)** daily temperature fields (GCM and PGFv3).  
     - Historical **quantile maps** derived from these fields on a fixed physical range.
   - A Vision Transformer learns to map the **Restormer-output HR quantile map** to a **reference-based HR quantile map** at each grid cell.:contentReference[oaicite:2]{index=2}  
   - Future GCM projections (SSP2-4.5, SSP5-8.5) are corrected using **rank-preserving, equidistant CDF matching** with the learned quantile transfer function.:contentReference[oaicite:3]{index=3}  

This design separates **spatial restoration** and **distributional alignment**, providing transparent control over distribution tails while maintaining an efficient end-to-end workflow.

---

## 3. Repository structure

```text
restormer_stage/
  Rs2.py               # Training script for Restormer-based restoration (64×64 patches)
  restormer_arch.py    # Official Restormer implementation (Zamir et al., 2021/2022)

vit_bias_correction/
  utils.py             # Quantile map generation (GPU-accelerated, fixed physical range)
  model.py             # SimpleViT architecture for bias correction
  train.py             # Training utilities and fixed-range training pipeline
  main.py              # Full pipeline: data preparation → training → SSP inference
  inference.py         # Bias-correction inference utilities
  run.ipynb            # Example notebook (optional)

restormer_stage/Rs2.py trains the Restormer on daily Tmax/Tas/Tmin using 64×64 patches, saving model checkpoints and PSNR/SSIM diagnostics.
vit_bias_correction/utils.py computes GPU-accelerated fixed-range quantile maps and manages caching.
vit_bias_correction/model.py defines the ViT block and SimpleViT model with optional sigmoid output to constrain quantile maps to [0, 1].
vit_bias_correction/train.py provides training loops for fixed-range and standard modes, including default hyperparameter configurations.

vit_bias_correction/main.py implements a three-step pipeline:
1. Prepare historical quantile maps and cache them.
2. Train the ViT model.
3. Apply bias correction to multiple future SSP scenarios.

## 4. Requirements

The code is implemented in Python with PyTorch. Main dependencies include:
python >= 3.9
pytorch >= 1.12 (GPU recommended)
xarray
netCDF4
numpy
tqdm
matplotlib
scipy
einops

## 5. Data

BHRR is designed for daily near-surface air temperature from CMIP6 GCMs and PGFv3:
CMIP6 GCM: ACCESS-CM2 daily Tas/Tmax/Tmin (historical and SSP scenarios).
Reference: Princeton Global Forcing v3 (PGFv3) 0.25° daily Tas/Tmax/Tmin.
Benchmark: NEX-GDDP-CMIP6 downscaled projections for comparison (not required for training).

Data availability:
CMIP6 GCM: ESGF (e.g., CEDA node).
NEX-GDDP-CMIP6: NASA NEX GDDP CMIP6 archive.
PGFv3: Princeton Global Forcing v3 dataset (Hydrology group, University of Southampton).
