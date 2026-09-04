# sar2optical_train.py (Best Config - 2025-09-03)
# SAR-to-Optical Self-attention Res-UNet for Grayscale Images Colorization Implementation Code
# Ahmed M. Abdelaziz
# Ahmed.Hussien5@student.aast.edu
# AASTMT


# =======================================================================================
# S2O-SARUNet - Training Configuration
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



import os, glob, csv, random, time
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from pytorch_msssim import ssim as torch_ssim
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

from sar2optical_model import GeneratorUNet, Discriminator, VGGPerceptualLoss

# ===================== CONFIG =====================
SAR_PATTERN = r"C:/S2O/SEN12/s1_*"
OPTICAL_PATTERN = r"C:/S2O/SEN12/s2_*"
OUTPUT_DIR = Path(r"C:/S2O/SEN12/training_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.20, 0.10
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 15
MIN_DELTA = 0.001
BATCH_SIZE = 4
NUM_WORKERS = 0
G_LR, D_LR, MIN_LR = 2e-4, 1e-4, 1e-6
BETAS = (0.5, 0.999)
LAMBDA_ADV, LAMBDA_L1, LAMBDA_SSIM, LAMBDA_PERC = 1.0, 10.0, 5.0, 0.1
REAL_LABEL, FAKE_LABEL, G_REAL_LABEL = 0.9, 0.0, 1.0

# 1 = calculate LPIPS/FID/SAM/ERGAS every epoch.
# If validation becomes too slow, change this to 5 or 10.
FULL_METRICS_EVERY = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

print("\n" + "=" * 78)
print("S2O-SARUNet Training")
print("Metrics: SSIM | PSNR | LPIPS | FID | SAM | ERGAS")
print("=" * 78)
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"CUDA   : {torch.version.cuda}")
print("=" * 78 + "\n")

# ===================== DATA =====================
def discover_pairs(sar_pattern, optical_pattern):
    sar_dirs = sorted(Path(p) for p in glob.glob(sar_pattern) if Path(p).is_dir())
    opt_dirs = sorted(Path(p) for p in glob.glob(optical_pattern) if Path(p).is_dir())
    if not sar_dirs or not opt_dirs:
        raise FileNotFoundError("Check SAR_PATTERN and OPTICAL_PATTERN.")

    optical_index = {}
    for od in opt_dirs:
        for p in od.rglob("*.png"):
            optical_index[p.name] = p

    pairs = []
    for sd in sar_dirs:
        for s in sd.rglob("*.png"):
            if "_s1_" in s.name:
                o = optical_index.get(s.name.replace("_s1_", "_s2_"))
                if o is not None:
                    pairs.append((s, o))

    if not pairs:
        raise RuntimeError("No valid PNG SAR/optical pairs found.")

    return sorted(pairs, key=lambda z: str(z[0]))


def split_pairs(pairs):
    pairs = list(pairs)
    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    ntr = int(n * TRAIN_RATIO)
    nv = int(n * VAL_RATIO)
    return pairs[:ntr], pairs[ntr:ntr + nv], pairs[ntr + nv:]


def save_manifest(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sar_path", "optical_path"])
        for a, b in pairs:
            w.writerow([str(a), str(b)])


class SAROpticalDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.sar_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        self.opt_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, o = self.pairs[i]
        with Image.open(s) as im:
            sar = self.sar_tf(im.convert("L"))
        with Image.open(o) as im:
            opt = self.opt_tf(im.convert("RGB"))
        return sar, opt


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ===================== METRICS =====================
def calc_sam(pred, target, eps=1e-8):
    """RGB SAM in degrees. Lower is better."""
    p = pred.reshape(-1, 3).astype(np.float64)
    t = target.reshape(-1, 3).astype(np.float64)

    pn = np.linalg.norm(p, axis=1)
    tn = np.linalg.norm(t, axis=1)
    valid = (pn > eps) & (tn > eps)

    if not np.any(valid):
        return np.nan

    p, t = p[valid], t[valid]
    pn, tn = pn[valid], tn[valid]
    dot = np.sum(p * t, axis=1)
    cosang = np.clip(dot / (pn * tn + eps), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)).mean())


def calc_ergas(pred, target, ratio=1.0, eps=1e-8):
    """ERGAS for same-resolution RGB images. Lower is better."""
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=(0, 1)))
    mean_ref = np.maximum(np.abs(np.mean(t, axis=(0, 1))), eps)
    return float((100.0 / ratio) * np.sqrt(np.mean((rmse / mean_ref) ** 2)))


def image_ssim_psnr(pred, target):
    s = structural_similarity(target, pred, channel_axis=2, data_range=1.0)
    p = peak_signal_noise_ratio(target, pred, data_range=1.0)
    return float(s), float(p)


@torch.no_grad()
def evaluate(G, loader, lpips_metric, full_metrics=True):
    G.eval()

    l1_vals, ssim_vals, psnr_vals = [], [], []
    lpips_vals, sam_vals, ergas_vals = [], [], []

    fid_metric = None
    if full_metrics:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
        fid_metric.reset()

    for sar, opt in tqdm(loader, desc="Validation", leave=False):
        sar = sar.to(DEVICE, non_blocking=True)
        opt = opt.to(DEVICE, non_blocking=True)

        fake = G(sar)

        # Validation L1 in native [-1,1] space.
        batch_l1 = torch.mean(torch.abs(fake - opt), dim=(1, 2, 3))
        l1_vals.extend(batch_l1.cpu().numpy().tolist())

        fake01 = to_01(fake)
        opt01 = to_01(opt)

        fn = fake01.cpu().numpy()
        on = opt01.cpu().numpy()

        for i in range(fn.shape[0]):
            pred = np.transpose(fn[i], (1, 2, 0))
            ref = np.transpose(on[i], (1, 2, 0))

            s, p = image_ssim_psnr(pred, ref)
            ssim_vals.append(s)
            psnr_vals.append(p)

            if full_metrics:
                sam_vals.append(calc_sam(pred, ref))
                ergas_vals.append(calc_ergas(pred, ref, ratio=1.0))

        if full_metrics:
            # LPIPS expects [-1,1], which matches fake and opt.
            lp = lpips_metric(fake, opt).reshape(-1)
            lpips_vals.extend(lp.cpu().numpy().tolist())

            # FID is dataset-level and updated batch-by-batch.
            fid_metric.update(opt01, real=True)
            fid_metric.update(fake01, real=False)

    result = {
        "l1": float(np.mean(l1_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "lpips": np.nan,
        "fid": np.nan,
        "sam": np.nan,
        "ergas": np.nan,
    }

    if full_metrics:
        result["lpips"] = float(np.nanmean(lpips_vals))
        result["sam"] = float(np.nanmean(sam_vals))
        result["ergas"] = float(np.nanmean(ergas_vals))
        if len(loader.dataset) >= 2:
            result["fid"] = float(fid_metric.compute().item())

        del fid_metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ===================== LOAD DATA =====================
pairs = discover_pairs(SAR_PATTERN, OPTICAL_PATTERN)
train_pairs, val_pairs, test_pairs = split_pairs(pairs)

print(
    f"Total={len(pairs)} | Train={len(train_pairs)} | "
    f"Val={len(val_pairs)} | Test={len(test_pairs)}"
)

save_manifest(train_pairs, OUTPUT_DIR / "split_training_70.csv")
save_manifest(val_pairs, OUTPUT_DIR / "split_validation_20.csv")
save_manifest(test_pairs, OUTPUT_DIR / "split_testing_10.csv")

train_loader = DataLoader(
    SAROpticalDataset(train_pairs),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    SAROpticalDataset(val_pairs),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ===================== MODEL / LOSS =====================
G = GeneratorUNet().to(DEVICE)
D = Discriminator().to(DEVICE)

adv_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
vgg_loss = VGGPerceptualLoss().to(DEVICE)
vgg_loss.eval()
for p in vgg_loss.parameters():
    p.requires_grad = False

print("Loading LPIPS...")
lpips_metric = lpips.LPIPS(net="alex").to(DEVICE)
lpips_metric.eval()
for p in lpips_metric.parameters():
    p.requires_grad = False
print("LPIPS ready.\n")

opt_G = torch.optim.Adam(G.parameters(), lr=G_LR, betas=BETAS)
opt_D = torch.optim.Adam(D.parameters(), lr=D_LR, betas=BETAS)

sch_G = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_G, T_max=MAX_EPOCHS, eta_min=MIN_LR
)
sch_D = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_D, T_max=MAX_EPOCHS, eta_min=MIN_LR
)

history = {
    "g_loss": [], "d_loss": [],
    "train_ssim": [], "train_psnr": [],
    "val_l1": [], "val_ssim": [], "val_psnr": [],
    "val_lpips": [], "val_fid": [], "val_sam": [], "val_ergas": [],
    "g_lr": [], "d_lr": [],
    "train_minutes": [], "val_minutes": []
}

best_ssim = -1.0
no_improve = 0
best_path = OUTPUT_DIR / "checkpoint_best.pth"
generator_best_path = OUTPUT_DIR / "generator_best.pth"
overall_start = time.perf_counter()

# ===================== TRAIN =====================
for epoch in range(1, MAX_EPOCHS + 1):
    train_start = time.perf_counter()

    G.train()
    D.train()

    sum_g = 0.0
    sum_d = 0.0
    train_ssim_sum = 0.0
    train_psnr_sum = 0.0
    train_count = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{MAX_EPOCHS}")

    for sar, opt in loop:
        sar = sar.to(DEVICE, non_blocking=True)
        opt = opt.to(DEVICE, non_blocking=True)

        # --------------------- DISCRIMINATOR ---------------------
        # cGAN CONDITION: Discriminator internally concatenates
        # SAR (1 channel) + optical (3 channels) = 4 channels.
        with torch.no_grad():
            fake_det = G(sar)

        d_real = D(sar, opt)
        d_fake = D(sar, fake_det)

        dloss = 0.5 * (
            adv_loss(d_real, torch.full_like(d_real, REAL_LABEL)) +
            adv_loss(d_fake, torch.full_like(d_fake, FAKE_LABEL))
        )

        opt_D.zero_grad(set_to_none=True)
        dloss.backward()
        opt_D.step()

        # ----------------------- GENERATOR -----------------------
        fake = G(sar)
        pred = D(sar, fake)

        gadv = adv_loss(pred, torch.full_like(pred, G_REAL_LABEL))
        gl1 = l1_loss(fake, opt)
        gssim = 1.0 - torch_ssim(
            to_01(fake), to_01(opt), data_range=1.0, size_average=True
        )
        gperc = vgg_loss(fake, opt)

        gloss = (
            LAMBDA_ADV * gadv +
            LAMBDA_L1 * gl1 +
            LAMBDA_SSIM * gssim +
            LAMBDA_PERC * gperc
        )

        opt_G.zero_grad(set_to_none=True)
        gloss.backward()
        opt_G.step()

        sum_g += gloss.item()
        sum_d += dloss.item()

        # Train SSIM/PSNR from current fake batch (no extra G pass).
        fn = to_01(fake.detach()).cpu().numpy()
        on = to_01(opt.detach()).cpu().numpy()

        for i in range(fn.shape[0]):
            pred_np = np.transpose(fn[i], (1, 2, 0))
            ref_np = np.transpose(on[i], (1, 2, 0))
            s, p = image_ssim_psnr(pred_np, ref_np)
            train_ssim_sum += s
            train_psnr_sum += p
            train_count += 1

        running_ssim = train_ssim_sum / train_count
        running_psnr = train_psnr_sum / train_count

        loop.set_postfix(
            D=f"{dloss.item():.4f}",
            G=f"{gloss.item():.4f}",
            PSNR=f"{running_psnr:.2f}",
            SSIM=f"{running_ssim:.4f}"
        )

    train_minutes = (time.perf_counter() - train_start) / 60.0

    avg_g = sum_g / len(train_loader)
    avg_d = sum_d / len(train_loader)
    avg_train_ssim = train_ssim_sum / train_count
    avg_train_psnr = train_psnr_sum / train_count

    # ---------------------- VALIDATION ----------------------
    val_start = time.perf_counter()

    run_full_metrics = (
        epoch == 1 or
        epoch % FULL_METRICS_EVERY == 0 or
        epoch == MAX_EPOCHS
    )

    vm = evaluate(
        G,
        val_loader,
        lpips_metric,
        full_metrics=run_full_metrics
    )

    val_minutes = (time.perf_counter() - val_start) / 60.0

    sch_G.step()
    sch_D.step()

    # ------------------------ HISTORY ------------------------
    history["g_loss"].append(avg_g)
    history["d_loss"].append(avg_d)
    history["train_ssim"].append(avg_train_ssim)
    history["train_psnr"].append(avg_train_psnr)
    history["val_l1"].append(vm["l1"])
    history["val_ssim"].append(vm["ssim"])
    history["val_psnr"].append(vm["psnr"])
    history["val_lpips"].append(vm["lpips"])
    history["val_fid"].append(vm["fid"])
    history["val_sam"].append(vm["sam"])
    history["val_ergas"].append(vm["ergas"])
    history["g_lr"].append(opt_G.param_groups[0]["lr"])
    history["d_lr"].append(opt_D.param_groups[0]["lr"])
    history["train_minutes"].append(train_minutes)
    history["val_minutes"].append(val_minutes)

    # ------------------ REQUESTED CONSOLE STYLE ------------------
    print("\n" + "-" * 78)
    print(f"Epoch {epoch:03d}")
    print(f"Train G Loss : {avg_g:.4f}")
    print(f"Train D Loss : {avg_d:.4f}")
    print(f"Train SSIM   : {avg_train_ssim:.4f}")
    print(f"Train PSNR   : {avg_train_psnr:.2f} dB")
    print(f"Val L1       : {vm['l1']:.4f}")
    print(f"Val SSIM     : {vm['ssim']:.4f}")
    print(f"Val PSNR     : {vm['psnr']:.2f} dB")

    if run_full_metrics:
        print(f"Val LPIPS    : {vm['lpips']:.4f}")
        print(f"Val FID      : {vm['fid']:.4f}")
        print(f"Val SAM      : {vm['sam']:.4f} deg")
        print(f"Val ERGAS    : {vm['ergas']:.4f}")
    else:
        print(f"Val LPIPS    : skipped (every {FULL_METRICS_EVERY} epochs)")
        print(f"Val FID      : skipped (every {FULL_METRICS_EVERY} epochs)")
        print(f"Val SAM      : skipped (every {FULL_METRICS_EVERY} epochs)")
        print(f"Val ERGAS    : skipped (every {FULL_METRICS_EVERY} epochs)")

    print(f"Train time   : {train_minutes:.1f} min")
    print(f"Val time     : {val_minutes:.1f} min")
    print("-" * 78)

    # ---------------- CHECKPOINT / EARLY STOP ----------------
    if vm["ssim"] > best_ssim + MIN_DELTA:
        best_ssim = vm["ssim"]
        no_improve = 0

        torch.save({
            "epoch": epoch,
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "best_val_ssim": best_ssim,
            "val_psnr": vm["psnr"],
            "val_lpips": vm["lpips"],
            "val_fid": vm["fid"],
            "val_sam": vm["sam"],
            "val_ergas": vm["ergas"],
            "seed": SEED
        }, best_path)

        torch.save(G.state_dict(), generator_best_path)

        print(
            f"BEST CHECKPOINT SAVED | Epoch={epoch} | "
            f"Val SSIM={best_ssim:.4f}"
        )
    else:
        no_improve += 1
        print(f"No SSIM improvement: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print("EARLY STOPPING.")
            break

# ===================== SAVE HISTORY =====================
np.savez(
    OUTPUT_DIR / "training_history.npz",
    **{k: np.asarray(v, dtype=float) for k, v in history.items()}
)

epoch_axis = np.arange(1, len(history["g_loss"]) + 1)


def save_curve(values, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, np.asarray(values, dtype=float))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


for key, title, ylabel, filename in [
    ("g_loss", "Generator Loss", "Loss", "curve_generator_loss.png"),
    ("d_loss", "Discriminator Loss", "Loss", "curve_discriminator_loss.png"),
    ("train_ssim", "Training SSIM", "SSIM", "curve_train_ssim.png"),
    ("train_psnr", "Training PSNR", "PSNR (dB)", "curve_train_psnr.png"),
    ("val_ssim", "Validation SSIM", "SSIM", "curve_val_ssim.png"),
    ("val_psnr", "Validation PSNR", "PSNR (dB)", "curve_val_psnr.png"),
    ("val_lpips", "Validation LPIPS", "LPIPS", "curve_val_lpips.png"),
    ("val_fid", "Validation FID", "FID", "curve_val_fid.png"),
    ("val_sam", "Validation SAM", "SAM (degrees)", "curve_val_sam.png"),
    ("val_ergas", "Validation ERGAS", "ERGAS", "curve_val_ergas.png"),
]:
    save_curve(history[key], title, ylabel, filename)

# Learning-rate curve
plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, history["g_lr"], label="G LR")
plt.plot(epoch_axis, history["d_lr"], label="D LR")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curve_learning_rate.png", dpi=300)
plt.close()

# Combined train/validation SSIM
plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, history["train_ssim"], label="Train SSIM")
plt.plot(epoch_axis, history["val_ssim"], label="Val SSIM")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("Train vs Validation SSIM")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curve_ssim_train_vs_val.png", dpi=300)
plt.close()

# Combined train/validation PSNR
plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, history["train_psnr"], label="Train PSNR")
plt.plot(epoch_axis, history["val_psnr"], label="Val PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Train vs Validation PSNR")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curve_psnr_train_vs_val.png", dpi=300)
plt.close()

# ===================== FINAL SUMMARY =====================
total_hours = (time.perf_counter() - overall_start) / 3600.0

print("\n" + "=" * 78)
print("TRAINING COMPLETED")
print("=" * 78)
print(f"Epochs completed : {len(history['g_loss'])}")
print(f"Best Val SSIM    : {best_ssim:.4f}")
print(f"Total time       : {total_hours:.2f} h")
print(f"Best checkpoint  : {best_path}")
print(f"Generator only   : {generator_best_path}")
print("=" * 78)
