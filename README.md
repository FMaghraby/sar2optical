# SAR-to-Optical Self-Attention ResUNet for Grayscale Image Colorization

## Abstract

Synthetic Aperture Radar (SAR) imagery provides all-weather, day-and-night Earth observation capabilities but lacks the spectral richness of optical imagery, which can limit visual interpretability. This project implements *S2O-SARUNet, a conditional Generative Adversarial Network (cGAN) for SAR-to-Optical (S2O) image translation using the **SEN1-2 dataset*.

The generator follows a *U-Net encoder-decoder architecture* with multi-scale skip connections, *single-head spatial self-attention (SHSA)* at the bottleneck, and *four residual blocks* for feature refinement. A *spectrally normalized conditional PatchGAN discriminator* provides patch-level adversarial supervision. Training incorporates *L1 reconstruction loss, SSIM loss, VGG-based perceptual loss, cosine-annealed learning rates, and early stopping*.

The reported evaluation achieved an average *SSIM of 0.770* and *PSNR of 25.85 dB, with complementary **LPIPS = 0.302, FID = 114.67, SAM = 8.60°, and ERGAS = 36.59*. The study also evaluates baseline cGAN, transformer-augmented, and diffusion-refinement configurations.

## Implementation Details

•⁠  ⁠*Architecture:* Conditional GAN (cGAN)
•⁠  ⁠*Generator:* U-Net + SHSA + 4 bottleneck residual blocks
•⁠  ⁠*Discriminator:* Spectrally normalized conditional PatchGAN
•⁠  ⁠*Input:* SAR grayscale image
•⁠  ⁠*Output:* 3-channel RGB optical image
•⁠  ⁠*Image resolution:* 128 × 128
•⁠  ⁠*Dataset split:* 70% training / 20% validation / 10% testing
•⁠  ⁠*Batch size:* 4
•⁠  ⁠*Optimizer:* Adam (⁠ β1 = 0.5 ⁠, ⁠ β2 = 0.999 ⁠)
•⁠  ⁠*Learning rate:* ⁠ 2e-4 ⁠ with cosine annealing
•⁠  ⁠*Maximum epochs:* 300
•⁠  ⁠*Early stopping:* 15 epochs without sufficient validation SSIM improvement
•⁠  ⁠*Hardware:* NVIDIA RTX A2000 (4 GB VRAM), Intel Core i7-11850H, 32 GB RAM
•⁠  ⁠*Primary validation metrics:* SSIM and PSNR
•⁠  ⁠*Additional evaluation metrics:* LPIPS, FID, SAM, and ERGAS

## Dataset

The project uses the *SEN1-2 SAR–Optical dataset*, containing geographically distributed SAR and optical image pairs acquired across different seasons.

*Dataset:*  
https://mediatum.ub.tum.de/1436631

Use password ⁠ m1436631 ⁠ when required.

The implementation expects PNG SAR and optical images and uses a fixed *70% / 20% / 10%* train-validation-test split.

## Model Overview

SAR Image  
→ Encoder  
→ Single-Head Spatial Self-Attention  
→ 4 Residual Blocks  
→ U-Net Decoder + Skip Connections  
→ Generated Optical Image

During training, the conditional PatchGAN discriminator receives:

⁠ [SAR, Real Optical] ⁠ or ⁠ [SAR, Generated Optical] ⁠

This conditioning explicitly guides the adversarial learning process using the input SAR image.
