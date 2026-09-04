# sar2optical_model.py (Best Config - 2025-09-03)
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


# =======================================================================================
# S2O-SARUNet - Model Configuration
# =======================================================================================
# Generator:          U-Net Generator + Single-Head Spatial Self-Attention (SHSA)
#                     + 4 Residual Blocks at the Bottleneck
#
# Discriminator:      Conditional PatchGAN with Spectral Normalization,
#                     Instance Normalization, LeakyReLU, and Dropout
#
# cGAN Condition:     Discriminator input = SAR + Real/Generated Optical image
#                     concatenated channel-wise
#
# Generator Loss:     1×Adversarial + 10×L1 + 5×SSIM
#                     + 0.1×Perceptual Loss (VGG16 features[:9], ReLU2_2)
#
# Dataset:            Paired PNG images discovered from s1_* ↔ s2_* directories
#                     using filename mapping _s1_ → _s2_
#
# Data Split:         70% Training / 20% Validation / 10% Testing
#
# Optimizer:          Adam
#                     Generator LR = 2e-4
#                     Discriminator LR = 1e-4
#                     Betas = (0.5, 0.999)
#
# LR Scheduler:       Cosine Annealing, minimum LR = 1e-6
#
# SSIM Loss:          pytorch_msssim SSIM implementation
#
# Early Stopping:     Validation SSIM, patience = 15, min_delta = 0.001
#
# Device Support:     CUDA-enabled; configured for NVIDIA RTX A2000 4 GB
#
# Generator Output:   3-channel RGB optical image with Tanh activation [-1, 1]
#
# Input Scaling:      SAR: [0, 1]
#                     Optical target: normalized to [-1, 1]
# =======================================================================================





import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head spatial self-attention at the bottleneck."""
    def __init__(self, in_dim=512):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)   # B,HW,C/8
        k = self.key(x).view(b, -1, h*w)                      # B,C/8,HW
        a = self.softmax(torch.bmm(q, k))                     # B,HW,HW
        v = self.value(x).view(b, -1, h*w)                    # B,C,HW
        out = torch.bmm(v, a.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorUNet(nn.Module):
    """True Residual U-Net + bottleneck SHSA for SAR->Optical translation."""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Encoder: 128->64->32->16->8
        self.down1 = down_block(in_channels, 64)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        # Bottleneck: global context + residual refinement
        self.attn = SelfAttention(512)
        self.res = nn.Sequential(*[ResidualBlock(512) for _ in range(4)])

        # Decoder with U-Net concatenation skips
        self.up1 = up_block(512, 256)  # concat E3 -> 512
        self.up2 = up_block(512, 128)  # concat E2 -> 256
        self.up3 = up_block(256, 64)   # concat E1 -> 128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)   # B,64,64,64
        d2 = self.down2(d1)  # B,128,32,32
        d3 = self.down3(d2)  # B,256,16,16
        d4 = self.down4(d3)  # B,512,8,8

        x = self.attn(d4)
        x = self.res(x)

        x = self.up1(x)
        x = torch.cat([x, d3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, d1], dim=1)
        return self.final(x)


class Discriminator(nn.Module):
    """Spectrally normalized conditional PatchGAN: SAR(1)+Optical(3)->15x15 map."""
    def __init__(self, in_channels=4):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
            )

        self.model = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, 1, 1)),
        )

    def forward(self, sar, optical):
        return self.model(torch.cat([sar, optical], dim=1))


class VGGPerceptualLoss(nn.Module):
    """Frozen ImageNet VGG16 features through ReLU2_2."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        y = ((y + 1.0) / 2.0).clamp(0, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.vgg(x), self.vgg(y))
