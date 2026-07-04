# sar2optical_train.py (Best Config - 2025-07-03)

# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


#20250703
#300 epochs
#SSIM + VGG11 + L1 + GAN Loss
#Learning rate schedule
#Early stopping
#Supports GPU with RTX A2000 (4GB)


# sar2optical_train.py
# sar2optical_train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pytorch_msssim import ssim
from sar2optical_model import GeneratorUNet, Discriminator, VGGPerceptualLoss
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim_fn
import numpy as np


# ---------------------------
# Safe Dataset Loader
# ---------------------------
class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.transform = transform
        self.samples = []
        for f in os.listdir(sar_dir):
            if f.endswith('.png') and '_s1_' in f:
                match = f.replace('_s1_', '_s2_')
                if os.path.exists(os.path.join(opt_dir, match)):
                    self.samples.append(f)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s1_name = self.samples[idx]
        s2_name = s1_name.replace('_s1_', '_s2_')
        sar = Image.open(os.path.join(self.sar_dir, s1_name)).convert("L")
        opt = Image.open(os.path.join(self.opt_dir, s2_name)).convert("RGB")
        if self.transform:
            sar = self.transform(sar)
            opt = self.transform(opt)
        return sar, opt

# ---------------------------
# Config
# ---------------------------
sar_dir = "C:/S2O/SEN12/s1"
opt_dir = "C:/S2O/SEN12/s2"
checkpoint_path = "checkpoint_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 300
lr = 0.0002
batch_size = 4
patience = 15

# ---------------------------
# Models
# ---------------------------
G = GeneratorUNet().to(device)

# Modified Discriminator (reduced depth + Dropout + InstanceNorm2d)
class StableDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
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
            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, padding=1))
        )
    def forward(self, sar, opt):
        x = torch.cat([sar, opt], dim=1)
        return self.model(x)

D = StableDiscriminator().to(device)

# Losses
vgg_loss = VGGPerceptualLoss().to(device)
l1_loss = nn.L1Loss()
adv_loss = nn.MSELoss()

# Optimizers
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs)

# Transform & Loader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = SAROpticalDataset(sar_dir, opt_dir, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# Training Loop
# ---------------------------
best_ssim, no_improve = 0, 0
losses, ssims, psnrs = [], [], []

for epoch in range(1, epochs + 1):
    G.train(); D.train()
    total_loss, total_ssim, total_psnr = 0, 0, 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
    for sar, opt in loop:
        sar, opt = sar.to(device), opt.to(device)
        fake = G(sar)

        # Discriminator
        valid = torch.ones_like(D(sar, opt))
        d_real = D(sar, opt)
        d_fake = D(sar, fake.detach())
        d_loss = 0.5 * (adv_loss(d_real, valid) + adv_loss(d_fake, torch.zeros_like(d_fake)))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # Generator
        pred_fake = D(sar, fake)
        g_adv = adv_loss(pred_fake, valid)
        g_l1 = l1_loss(fake, opt)
        g_perc = vgg_loss(fake, opt)
        g_ssim = 1 - ssim(fake, opt, data_range=1.0, size_average=True)
        g_loss = g_adv + 20 * g_l1 + 5 * g_ssim + 0.05 * g_perc

        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        # Metrics
        #psnr = 10 * torch.log10(1 / ((fake - opt) ** 2).mean()).item()
        #total_loss += g_loss.item()
        #total_ssim += 1 - g_ssim.item()
        #total_psnr += psnr

        #loop.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item(), PSNR=psnr, SSIM=1 - g_ssim.item())
        
        # Metrics
        # g_ssim used for loss (differentiable), eval_ssim for logging (skimage)
        g_ssim = 1 - ssim(fake, opt, data_range=1.0, size_average=True)

        # Use first image for reporting (eval only)
        eval_ssim = ssim_fn(fake[0].detach().permute(1, 2, 0).cpu().numpy(),
                    opt[0].detach().permute(1, 2, 0).cpu().numpy(),
                    channel_axis=2, data_range=1.0)

        psnr = 10 * torch.log10(1 / ((fake - opt) ** 2).mean()).item()

        total_loss += g_loss.item()
        total_ssim += eval_ssim
        total_psnr += psnr

        loop.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item(), PSNR=psnr, SSIM=eval_ssim)

        
        
        

    # Logging
    avg_ssim = total_ssim / len(loader)
    avg_psnr = total_psnr / len(loader)
    ssims.append(avg_ssim); psnrs.append(avg_psnr); losses.append(total_loss / len(loader))
    scheduler.step()

    # Save best
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        no_improve = 0
        torch.save(G.state_dict(), checkpoint_path)
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(losses); plt.title("Generator Loss")
plt.subplot(1, 2, 2); plt.plot(ssims, label="SSIM"); plt.plot(psnrs, label="PSNR"); plt.legend()
plt.tight_layout(); plt.show()
