# sar2optical_evaluate.py (Best Config - 2025-07-03)
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from sar2optical_model import GeneratorUNet
import numpy as np

# Settings
sar_dir = "C:/S2O/SEN12/test_s1_00"
opt_dir = "C:/S2O/SEN12/test_s2_00"
checkpoint_path = "checkpoint_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class EvalDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.transform = transform
        self.files = [f for f in os.listdir(sar_dir) if f.endswith('.png') and "_s1_" in f and 
                      os.path.exists(os.path.join(opt_dir, f.replace("_s1_", "_s2_")))]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        s1_name = self.files[idx]
        s2_name = s1_name.replace("_s1_", "_s2_")
        sar = Image.open(os.path.join(self.sar_dir, s1_name)).convert("L")
        opt = Image.open(os.path.join(self.opt_dir, s2_name)).convert("RGB")
        if self.transform:
            sar = self.transform(sar)
            opt = self.transform(opt)
        return sar, opt

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load model
G = GeneratorUNet().to(device)
G.load_state_dict(torch.load(checkpoint_path, map_location=device))
G.eval()

# Loader
dataset = EvalDataset(sar_dir, opt_dir, transform)
loader = DataLoader(dataset, batch_size=1)

# Evaluation
ssim_total, psnr_total = 0, 0
print("Evaluating on test set...")
for sar, gt in loader:
    sar, gt = sar.to(device), gt.to(device)
    with torch.no_grad():
        pred = G(sar)

    pred_np = pred[0].detach().cpu().permute(1, 2, 0).numpy()
    gt_np = gt[0].detach().cpu().permute(1, 2, 0).numpy()

    ssim_val = ssim_fn(pred_np, gt_np, data_range=1.0, channel_axis=2)
    psnr_val = 10 * torch.log10(1 / ((pred - gt) ** 2).mean()).item()

    ssim_total += ssim_val
    psnr_total += psnr_val

n = len(dataset)
print(f"\nFinal SSIM: {ssim_total / n:.4f}")
print(f"Final PSNR: {psnr_total / n:.2f} dB")
