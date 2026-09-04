# sar2optical_test.py (Best Config - 2025-09-03)
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


# =======================================================================================
# S2O-SARUNet - Validation and Statistical Evaluation
# =======================================================================================
# Purpose:
#   This script evaluates the trained S2O-SARUNet generator on the independent
#   20% validation partition saved during the 70/20/10 dataset split.
#
# Model:
#   U-Net Generator with Single-Head Spatial Self-Attention (SHSA) and
#   four Residual Blocks at the bottleneck.
#
# Evaluation Metrics:
#   - SSIM   : Structural similarity                          [Higher is better]
#   - PSNR   : Pixel-level reconstruction quality (dB)        [Higher is better]
#   - LPIPS  : Learned perceptual similarity                  [Lower is better]
#   - FID    : Distribution-level similarity                  [Lower is better]
#   - SAM    : RGB spectral-angle consistency (degrees)       [Lower is better]
#   - ERGAS  : RGB radiometric reconstruction error           [Lower is better]
#
# Statistical Analysis:
#   SSIM, PSNR, LPIPS, SAM, and ERGAS are calculated per validation image and
#   summarized using mean, standard deviation, min-max range, and 95% confidence
#   interval. FID is calculated over the complete validation-set distributions
#   and is therefore reported as a dataset-level point estimate.
#
# Validation Data:
#   Uses the saved 20% validation manifest generated during training to ensure
#   consistent evaluation using the same validation partition.
#
# Image Configuration:
#   SAR Input       : 1-channel grayscale, [0, 1]
#   Optical Target  : 3-channel RGB, normalized to [-1, 1]
#   Generated Image : 3-channel RGB, Tanh output [-1, 1]
#   Metric Range    : Generated/reference images converted to [0, 1] as required.
#
# Model Selection:
#   Loads the best checkpoint selected during training according to validation
#   SSIM improvement and the configured early-stopping criterion.
#
# Device:
#   Automatically uses CUDA when available; designed for NVIDIA RTX A2000 4 GB.
# =======================================================================================




import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from sar2optical_model import GeneratorUNet

OUTPUT_DIR = Path(r"C:/S2O/SEN12/training_output")
VAL_MANIFEST = OUTPUT_DIR / "split_validation_20.csv"
CHECKPOINT = OUTPUT_DIR / "checkpoint_best.pth"
IMAGE_SIZE = 128
BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(Path(r["sar_path"]), Path(r["optical_path"])) for r in csv.DictReader(f)]


class DS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.st = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.ot = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, o = self.pairs[i]
        with Image.open(s) as im:
            sar = self.st(im.convert("L"))
        with Image.open(o) as im:
            optical = self.ot(im.convert("RGB"))
        return sar, optical, str(s), str(o)


def to01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan,) * 6
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sd, float(v.min()), float(v.max()), mean - half, mean + half


def sam_deg(pred, ref, eps=1e-8):
    p = pred.reshape(-1, 3).astype(np.float64)
    r = ref.reshape(-1, 3).astype(np.float64)
    pn = np.linalg.norm(p, axis=1)
    rn = np.linalg.norm(r, axis=1)
    valid = (pn > eps) & (rn > eps)
    if not np.any(valid):
        return np.nan
    p, r, pn, rn = p[valid], r[valid], pn[valid], rn[valid]
    c = np.sum(p * r, axis=1) / (pn * rn + eps)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).mean())


def ergas(pred, ref, ratio=1.0, eps=1e-8):
    p = pred.astype(np.float64)
    r = ref.astype(np.float64)
    rmse = np.sqrt(np.mean((p - r) ** 2, axis=(0, 1)))
    mean_ref = np.maximum(np.abs(np.mean(r, axis=(0, 1))), eps)
    return float((100.0 / ratio) * np.sqrt(np.mean((rmse / mean_ref) ** 2)))


pairs = read_manifest(VAL_MANIFEST)
loader = DataLoader(
    DS(pairs), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

print(f"Device: {DEVICE}")
print(f"Validation pairs: {len(pairs)}")

G = GeneratorUNet().to(DEVICE)
ck = torch.load(CHECKPOINT, map_location=DEVICE)
G.load_state_dict(ck["generator"])
G.eval()

print("Loading LPIPS (AlexNet)...")
lpips_metric = lpips.LPIPS(net="alex").to(DEVICE).eval()
for p in lpips_metric.parameters():
    p.requires_grad = False

print("Loading FID (Inception-v3)...")
fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
fid_metric.reset()

ssim_vals, psnr_vals, lpips_vals, sam_vals, ergas_vals = [], [], [], [], []
rows = []

with torch.no_grad():
    for sar, opt, sar_paths, opt_paths in tqdm(loader, desc="Validation"):
        sar = sar.to(DEVICE, non_blocking=True)
        opt = opt.to(DEVICE, non_blocking=True)
        fake = G(sar)

        fake01 = to01(fake)
        opt01 = to01(opt)

        # LPIPS expects RGB tensors in [-1,1].
        lp_batch = lpips_metric(fake, opt).view(-1)

        # FID is a dataset-level distribution metric.
        fid_metric.update(opt01, real=True)
        fid_metric.update(fake01, real=False)

        fn = fake01.cpu().numpy()
        on = opt01.cpu().numpy()

        for i in range(fn.shape[0]):
            pred = np.transpose(fn[i], (1, 2, 0))
            ref = np.transpose(on[i], (1, 2, 0))

            ssim_v = structural_similarity(ref, pred, channel_axis=2, data_range=1.0)
            psnr_v = peak_signal_noise_ratio(ref, pred, data_range=1.0)
            lpips_v = float(lp_batch[i].item())
            sam_v = sam_deg(pred, ref)
            ergas_v = ergas(pred, ref, ratio=1.0)

            ssim_vals.append(ssim_v)
            psnr_vals.append(psnr_v)
            lpips_vals.append(lpips_v)
            sam_vals.append(sam_v)
            ergas_vals.append(ergas_v)
            rows.append([sar_paths[i], opt_paths[i], ssim_v, psnr_v, lpips_v, sam_v, ergas_v])

fid_value = float(fid_metric.compute().item())
metrics = {
    "SSIM": stats(ssim_vals),
    "PSNR": stats(psnr_vals),
    "LPIPS": stats(lpips_vals),
    "SAM": stats(sam_vals),
    "ERGAS": stats(ergas_vals),
}

print("\n" + "=" * 104)
print("VALIDATION-SET STATISTICAL EVALUATION")
print("=" * 104)
print(f"{'Metric':<10}{'Average':>14}{'Std':>14}{'Min':>14}{'Max':>14}{'95% CI':>32}")
print("-" * 104)
for name, r in metrics.items():
    mean, sd, mn, mx, lo, hi = r
    suffix = " dB" if name == "PSNR" else (" deg" if name == "SAM" else "")
    ci = f"[{lo:.4f}, {hi:.4f}]"
    print(f"{name:<10}{mean:>14.4f}{sd:>14.4f}{mn:>14.4f}{mx:>14.4f}{ci:>32}{suffix}")
print("-" * 104)
print(f"{'FID':<10}{fid_value:>14.4f}   (dataset-level point estimate)")
print("=" * 104)
print("Note: SSIM, PSNR, LPIPS, SAM, and ERGAS statistics are per-image. ")
print("      FID is computed once over the full validation distributions, so per-image avg/std/min-max/CI95 are not defined.")

results_path = OUTPUT_DIR / "validation_results_all_metrics.csv"
with open(results_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sar_path", "optical_path", "ssim", "psnr_db", "lpips", "sam_deg", "ergas"])
    w.writerows(rows)

summary_path = OUTPUT_DIR / "validation_summary_all_metrics.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "average", "std", "min", "max", "ci95_lower", "ci95_upper"])
    for name, r in metrics.items():
        w.writerow([name, *r])
    w.writerow(["FID", fid_value, "", "", "", "", ""])

print("Saved per-image results :", results_path)
print("Saved summary           :", summary_path)
