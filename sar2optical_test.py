# sar2optical_test.py (ResU-Net + Transformer/Self-Attention) - Best Config - 2025-07-03
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code

# Inference on the test set using the trained Generator; 
# Computes SSIM and PSNR for each sample
# Saves comparison visuals: SAR | Generated Optical | Ground Truth; Aggregates and prints average metrics
# Inference script with SSIM, PSNR, FID metrics and side-by-side image output


# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_score, peak_signal_noise_ratio as psnr_score
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from sar2optical_model import GeneratorUNet
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# === Directories ===
SAR_TEST_DIR = "C:/S2O/SEN12/s1_2k"
OPT_TEST_DIR = "C:/S2O/SEN12/s2_2k"
OUTPUT_DIR = "test_output_20250708"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Fix OpenMP Duplicate Library Error (Windows specific) ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Dataset Definition ===
class SAR2OpticalTestDataset(Dataset):
    def __init__(self, sar_dir, opt_dir):
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.sar_files = sorted([f for f in os.listdir(sar_dir) if f.endswith(".png")])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_file = self.sar_files[idx]
        opt_file = sar_file.replace("_s1_", "_s2_")
        sar_img = Image.open(os.path.join(self.sar_dir, sar_file)).convert("L")
        opt_img = Image.open(os.path.join(self.opt_dir, opt_file)).convert("RGB")
        sar_tensor = self.transform(sar_img)
        opt_tensor = self.transform(opt_img)
        return {
            'sar_tensor': sar_tensor,
            'opt_tensor': opt_tensor,
            'sar_img': sar_img,
            'opt_img': opt_img,
            'filename': sar_file
        }

# === Custom Collate Function ===
def custom_collate(batch):
    sar_tensor = torch.stack([item['sar_tensor'] for item in batch])
    opt_tensor = torch.stack([item['opt_tensor'] for item in batch])
    sar_img = [item['sar_img'] for item in batch]
    opt_img = [item['opt_img'] for item in batch]
    name = [item['filename'] for item in batch]
    return sar_tensor, opt_tensor, sar_img, opt_img, name

# === Load Trained Generator ===
model = GeneratorUNet().to(device)
model.load_state_dict(torch.load("checkpoint_best.pth", map_location=device))
model.eval()

# === DataLoader ===
dataset = SAR2OpticalTestDataset(SAR_TEST_DIR, OPT_TEST_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

# === Inception Model for FID ===
inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()
extractor = create_feature_extractor(inception, return_nodes={"avgpool": "features"})

# === Metric Accumulators ===
ssim_total, psnr_total = 0, 0
real_features, gen_features = [], []

# === Inference ===
with torch.no_grad():
    for i, (sar_tensor, opt_tensor, sar_img, opt_img, name) in enumerate(loader):
        sar_tensor = sar_tensor.to(device)
        opt_tensor = opt_tensor.to(device)

        # === Generate Prediction ===
        generated = model(sar_tensor).clamp(0, 1)
        real = opt_tensor

        # === Compute SSIM & PSNR ===
        gen_np = generated[0].cpu().permute(1, 2, 0).numpy()
        opt_np = real[0].cpu().permute(1, 2, 0).numpy()
        ssim = ssim_score(opt_np, gen_np, data_range=1.0, channel_axis=2)
        psnr = psnr_score(opt_np, gen_np, data_range=1.0)
        ssim_total += ssim
        psnr_total += psnr

        # === Save Side-by-Side Image ===
        sar_resized = np.array(sar_img[0].resize((128, 128)))
        gen_img = (gen_np * 255).astype(np.uint8)
        opt_img_np = (opt_np * 255).astype(np.uint8)
        sar_rgb = np.stack([sar_resized] * 3, axis=-1)
        combined = np.hstack([sar_rgb, gen_img, opt_img_np])
        Image.fromarray(combined).save(os.path.join(OUTPUT_DIR, f"compare_{name[0]}"), format="PNG")

        # === Save Visualized Plot ===
        plt.figure(figsize=(12, 4))
        plt.imshow(combined)
        plt.axis('off')
        plt.title(f"Generated Optical\nSSIM={ssim:.4f}, PSNR={psnr:.2f} dB", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{i+1}.png"))
        plt.close()

        # === Extract Features for FID ===
        real_resized = torch.nn.functional.interpolate(real, size=(299, 299), mode='bilinear', align_corners=False)
        gen_resized = torch.nn.functional.interpolate(generated, size=(299, 299), mode='bilinear', align_corners=False)
        real_feat = extractor(real_resized)['features'].view(real_resized.size(0), -1).cpu().numpy()
        gen_feat = extractor(gen_resized)['features'].view(gen_resized.size(0), -1).cpu().numpy()
        real_features.append(real_feat)
        gen_features.append(gen_feat)

# === FID Calculation ===
def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(np.array(act1).T)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(np.array(act2).T)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

# === Final Metrics ===
fid = calculate_fid(real_features, gen_features) if real_features and gen_features else float("nan")
avg_ssim = ssim_total / len(dataset)
avg_psnr = psnr_total / len(dataset)

print(f"\n Average SSIM: {avg_ssim:.4f}")
print(f" Average PSNR: {avg_psnr:.2f} dB")
#print(f" FID Score: {fid:.2f}")