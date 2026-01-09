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
