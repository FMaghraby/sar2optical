# sar2optical_model.py (Best Config - 2025-07-03)
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT




#20250703
#Improvements Applied

#Generator:	            Residual U-Net + Single-head/Self-Attention
#Discriminator:	            PatchGAN with SpectralNorm
#Losses:	            Adversarial + 10×L1 + 0.1×Perceptual (VGG16 relu2_2) + 5×SSIM
#Dataset:	            Directly loaded from folders s1_ ↔ s2_ using PNGs
#Optimizer:	            Adam, lr = 0.0002, betas = (0.5, 0.999)
#SSIM Loss:	            Uses pytorch_msssim for training stability
#Device Support:	    GPU-optimized for RTX A2000 4GB
#Output Activation:	    Tanh
#Input Normalization:	    [0, 1] range




import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(B, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.down1 = down_block(in_channels, 64)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        self.attn = SelfAttention(512)
        self.res = nn.Sequential(*[ResidualBlock(512) for _ in range(4)])

        self.up1 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        x = self.attn(d4)
        x = self.res(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final(x)


# ----------------------------------------
# Stable Discriminator (SpectralNorm + Dropout + InstanceNorm)
# ----------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):  # SAR (1) + Optical (3)
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2)
            )

        self.model = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, padding=1))  # reduced depth
        )

    def forward(self, sar, optical):
        x = torch.cat([sar, optical], dim=1)
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))