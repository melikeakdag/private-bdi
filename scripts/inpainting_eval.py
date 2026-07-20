# ============================================================
# Inpainting / Harmonizasyon Karsilastirma Notebooku
# HarmoniDiff vs RSPaint - Kapsamli Degerlendirme
# ============================================================
#
# Kullanim: Bu dosyayi Colab notebook'una yukleyip
#   %run inpainting_eval.py
# ile calistirabilir, ya da hucrelere boluk boluk yapistirabilirsin.
#
# Gereken paketler:
#   pip install scikit-image scipy pandas lpips torch --quiet
# (lpips ve torch opsiyonel; yoksa o metrik atlanir)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("UYARI: scikit-image yok, SSIM/PSNR atlanacak. `pip install scikit-image`")

try:
    import lpips
    import torch
    LPIPS_MODEL = lpips.LPIPS(net='alex')
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("UYARI: lpips/torch yok, LPIPS atlanacak. `pip install lpips torch`")


# ============================================================
# 0) AYARLAR - kendi yollarina gore duzenle
# ============================================================

ROOT = "/content/drive/MyDrive/SAR/private-bdi"

BG_DIR      = f"{ROOT}/outputs/synthetic_pairs"   # bg.png ve mask.png burada
HARMONI_DIR = f"{ROOT}/outputs/harmonized"        # result.png burada
RSPAINT_DIR = f"{ROOT}/outputs/rspaint_results"   # result_raw.png burada

MASK_FILENAME = "mask_location.png"          # instance klasorundeki mask dosyasi adi
BG_FILENAME = "bg.png"
HARMONI_RESULT_FILENAME = "result.png"
RSPAINT_RESULT_FILENAME = "result.png"


def discover_instances(bg_dir=BG_DIR, require_files=None):
    """BG_DIR altindaki tum instance klasorlerini otomatik bulur.

    require_files: (opsiyonel) her instance klasorunde bulunmasi gereken
    dosya adlari listesi. Verilirse, o dosyalar eksik olan klasorler atlanir
    (orn. henuz bg/mask uretilmemis klasorler).
    """
    if not os.path.isdir(bg_dir):
        raise FileNotFoundError(f"BG_DIR bulunamadi: {bg_dir}")

    all_entries = sorted(os.listdir(bg_dir))
    instances = []
    skipped = []

    for name in all_entries:
        full_path = os.path.join(bg_dir, name)
        if not os.path.isdir(full_path):
            continue
        if require_files:
            missing = [f for f in require_files if not os.path.exists(os.path.join(full_path, f))]
            if missing:
                skipped.append((name, missing))
                continue
        instances.append(name)

    print(f"BG_DIR'de {len(all_entries)} girdi bulundu, {len(instances)} tanesi instance klasoru olarak kabul edildi.")
    if skipped:
        print(f"{len(skipped)} klasor eksik dosya nedeniyle atlandi (ilk 5): {skipped[:5]}")

    return instances


# Tum dataset uzerinde calismak icin: bg.png ve mask dosyasi olan tum klasorleri bul
INSTANCES = discover_instances(BG_DIR, require_files=[BG_FILENAME, MASK_FILENAME])


# ============================================================
# 1) TEMEL YUKLEME / BOYUT UYUMLASTIRMA
# ============================================================

def load_rgb(path):
    """Herhangi bir goruntuyu (tif dahil) RGB float array olarak yukler."""
    return np.array(Image.open(path).convert("RGB")).astype(float)


def load_mask(path, binarize_thresh=127):
    """Mask'i tek kanal binary array olarak yukler (0/1)."""
    m = np.array(Image.open(path).convert("L"))
    return (m > binarize_thresh).astype(np.uint8)


def resize_to_match(img, target_hw):
    """img'i (H,W) hedefine gore yeniden boyutlandirir. img float veya uint8 olabilir."""
    h, w = target_hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    mode = "L" if img.ndim == 2 else "RGB"
    pil_img = Image.fromarray(img.astype(np.uint8), mode=mode) if img.ndim == 2 \
        else Image.fromarray(img.astype(np.uint8))
    resized = pil_img.resize((w, h))
    out = np.array(resized).astype(float if img.ndim == 3 else np.uint8)
    return out


def align_pair(img1, img2):
    """Iki goruntuyu ortak (kucuk olan) boyuta getirir."""
    if img1.shape[:2] == img2.shape[:2]:
        return img1, img2
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return resize_to_match(img1, (h, w)), resize_to_match(img2, (h, w))


# ============================================================
# 2) TEMEL FARK HARITASI
# ============================================================

def diff_map(img1, img2):
    img1, img2 = align_pair(img1, img2)
    return np.abs(img1 - img2).sum(axis=-1)


def stats_str(diff):
    return (f"max={diff.max():.0f}, ort={diff.mean():.2f}, "
            f"degisen(>10)=%{100 * (diff > 10).mean():.1f}")


# ============================================================
# 3) MASK-ICI vs MASK-DISI ANALIZ
# ============================================================

def masked_stats(diff, mask):
    """diff ve mask ayni boyutta olmali (mask gerekirse resize edilir)."""
    if diff.shape != mask.shape:
        mask = resize_to_match(mask, diff.shape)
        mask = (mask > 0.5).astype(np.uint8)
    inside = diff[mask > 0]
    outside = diff[mask == 0]
    return {
        "inside_mean": inside.mean() if inside.size else np.nan,
        "inside_std": inside.std() if inside.size else np.nan,
        "outside_mean": outside.mean() if outside.size else np.nan,
        "outside_leak_ratio": (outside.mean() / (inside.mean() + 1e-6)) if inside.size and outside.size else np.nan,
    }


# ============================================================
# 4) SINIR (BOUNDARY / DIKIS IZI) ANALIZI
# ============================================================

def boundary_ring(mask, width=5):
    """Mask kenarindan icten ve disten `width` piksel genisliginde bir halka doner."""
    dilated = ndimage.binary_dilation(mask, iterations=width)
    eroded = ndimage.binary_erosion(mask, iterations=width)
    return dilated & ~eroded.astype(bool)


def gradient_magnitude(img_rgb):
    gray = img_rgb.mean(axis=-1)
    gy, gx = np.gradient(gray)
    return np.sqrt(gx ** 2 + gy ** 2)


def boundary_discontinuity(bg, result, mask, ring_width=5):
    """Result'taki sinir bolgesindeki gradyan, bg'nin ayni bolgesine kiyasla ne kadar farkli."""
    bg, result = align_pair(bg, result)
    if mask.shape != bg.shape[:2]:
        mask = resize_to_match(mask, bg.shape[:2])
        mask = (mask > 0.5).astype(np.uint8)

    ring = boundary_ring(mask, width=ring_width)
    if ring.sum() == 0:
        return np.nan, np.nan

    grad_bg = gradient_magnitude(bg)
    grad_res = gradient_magnitude(result)

    bg_ring_grad = grad_bg[ring].mean()
    res_ring_grad = grad_res[ring].mean()
    # oran > 1 ise result sinirda bg'den daha "sert" gecis yapiyor -> dikis izi supheli
    return res_ring_grad, res_ring_grad / (bg_ring_grad + 1e-6)


# ============================================================
# 5) YUKSEK FREKANS (DOKU / ASIRI YUMUSAMA) ANALIZI
# ============================================================

def high_freq_energy(img_rgb, mask, low_freq_radius_ratio=8):
    """Mask alani icindeki yuksek frekans (yuksek detay) enerjisi."""
    gray = img_rgb.mean(axis=-1)
    if mask.shape != gray.shape:
        mask = resize_to_match(mask, gray.shape)
        mask = (mask > 0.5).astype(np.uint8)

    if mask.sum() == 0:
        return np.nan

    # yalnizca mask'in bounding-box'unda FFT al (daha anlamli lokal frekans)
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = gray[y0:y1, x0:x1]

    f = np.fft.fftshift(np.fft.fft2(patch))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = max(1, min(h, w) // low_freq_radius_ratio)

    high_freq = mag.copy()
    high_freq[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 0
    return high_freq.sum()


# ============================================================
# 6) RENK / HISTOGRAM TUTARLILIGI (HARMONIZASYON)
# ============================================================

def color_consistency(img, mask, ring_width=8):
    """Mask ici ortalama renk ile hemen disindaki (ring) ortalama renk arasindaki fark.
    Dusuk deger = iyi harmonizasyon (ton uyumu)."""
    if mask.shape != img.shape[:2]:
        mask = resize_to_match(mask, img.shape[:2])
        mask = (mask > 0.5).astype(np.uint8)

    ring = boundary_ring(mask, width=ring_width)
    if mask.sum() == 0 or ring.sum() == 0:
        return np.nan

    inside_mean_rgb = img[mask > 0].mean(axis=0)
    ring_mean_rgb = img[ring].mean(axis=0)
    return np.abs(inside_mean_rgb - ring_mean_rgb).sum()


# ============================================================
# 7) SSIM / PSNR / LPIPS
# ============================================================

def compute_ssim_psnr(img1, img2):
    if not HAS_SKIMAGE:
        return np.nan, np.nan
    img1, img2 = align_pair(img1, img2)
    a = img1.astype(np.uint8)
    b = img2.astype(np.uint8)
    s = ssim(a, b, channel_axis=-1)
    p = psnr(a, b)
    return s, p


def compute_lpips(img1, img2):
    if not HAS_LPIPS:
        return np.nan
    img1, img2 = align_pair(img1, img2)
    t1 = torch.tensor(img1 / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
    t2 = torch.tensor(img2 / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        d = LPIPS_MODEL(t1, t2).item()
    return d


# ============================================================
# 8) TEK ORNEK ICIN TAM ANALIZ + GORSELLESTIRME
# ============================================================

def full_compare(inst, show_plot=True):
    bg_path = f"{BG_DIR}/{inst}/{BG_FILENAME}"
    mask_path = f"{BG_DIR}/{inst}/{MASK_FILENAME}"
    harmoni_path = f"{HARMONI_DIR}/{inst}/{HARMONI_RESULT_FILENAME}"
    rspaint_path = f"{RSPAINT_DIR}/{inst}/{RSPAINT_RESULT_FILENAME}"

    bg = load_rgb(bg_path)
    mask = load_mask(mask_path)
    res1 = load_rgb(harmoni_path)   # HarmoniDiff
    res2 = load_rgb(rspaint_path)   # RSPaint

    row = {"instance": inst, "mask_area_px": int(mask.sum())}

    for name, res in [("harmoni", res1), ("rspaint", res2)]:
        d = diff_map(bg, res)
        mstats = masked_stats(d, mask)
        boundary_grad, boundary_ratio = boundary_discontinuity(bg, res, mask)
        hf_energy = high_freq_energy(res, mask)
        hf_energy_bg = high_freq_energy(bg, mask)
        color_diff = color_consistency(res, mask)
        s, p = compute_ssim_psnr(bg, res)
        lp = compute_lpips(bg, res)

        row.update({
            f"{name}_diff_mean": d.mean(),
            f"{name}_inside_mean": mstats["inside_mean"],
            f"{name}_outside_mean": mstats["outside_mean"],
            f"{name}_outside_leak_ratio": mstats["outside_leak_ratio"],
            f"{name}_boundary_grad": boundary_grad,
            f"{name}_boundary_ratio": boundary_ratio,   # >1 = dikis izi supheli
            f"{name}_hf_energy_ratio": hf_energy / (hf_energy_bg + 1e-6),  # <1 = asiri yumusama
            f"{name}_color_consistency": color_diff,     # dusuk = iyi harmonizasyon
            f"{name}_ssim": s,
            f"{name}_psnr": p,
            f"{name}_lpips": lp,
        })

    print(f"\n=== {inst} ===  (mask alani: {row['mask_area_px']} px)")
    for name in ["harmoni", "rspaint"]:
        print(f"  {name:8s}: diff_mean={row[f'{name}_diff_mean']:.2f} | "
              f"inside={row[f'{name}_inside_mean']:.2f} | "
              f"outside_leak={row[f'{name}_outside_leak_ratio']:.3f} | "
              f"boundary_ratio={row[f'{name}_boundary_ratio']:.2f} | "
              f"hf_ratio={row[f'{name}_hf_energy_ratio']:.2f} | "
              f"color_diff={row[f'{name}_color_consistency']:.2f} | "
              f"SSIM={row[f'{name}_ssim']:.3f} | PSNR={row[f'{name}_psnr']:.1f} | "
              f"LPIPS={row[f'{name}_lpips']:.3f}")

    if show_plot:
        d1 = diff_map(bg, res1)
        d2 = diff_map(bg, res2)
        vmax = max(d1.max(), d2.max())

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes[0, 0].imshow(bg.astype(np.uint8)); axes[0, 0].set_title("bg (oncesi)")
        axes[0, 1].imshow(mask, cmap="gray"); axes[0, 1].set_title("mask")
        axes[0, 2].imshow(res1.astype(np.uint8)); axes[0, 2].set_title("HarmoniDiff")
        axes[0, 3].imshow(res2.astype(np.uint8)); axes[0, 3].set_title("RSPaint")

        im1 = axes[1, 0].imshow(d1, cmap="hot", vmin=0, vmax=vmax); axes[1, 0].set_title("fark: HarmoniDiff")
        im2 = axes[1, 1].imshow(d2, cmap="hot", vmin=0, vmax=vmax); axes[1, 1].set_title("fark: RSPaint")

        ring = boundary_ring(resize_to_match(mask, d1.shape) if mask.shape != d1.shape else mask, width=5)
        ring_overlay = np.zeros((*ring.shape, 3), dtype=np.uint8)
        ring_overlay[ring] = [255, 0, 0]
        axes[1, 2].imshow(res1.astype(np.uint8)); axes[1, 2].imshow(ring_overlay, alpha=0.4)
        axes[1, 2].set_title("HarmoniDiff + sinir bolgesi")
        axes[1, 3].imshow(res2.astype(np.uint8)); axes[1, 3].imshow(ring_overlay, alpha=0.4)
        axes[1, 3].set_title("RSPaint + sinir bolgesi")

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)
        plt.suptitle(inst, fontsize=14)
        plt.tight_layout()
        plt.show()

    return row


# ============================================================
# 9) TUM DATASET UZERINDE CALISTIR + OZET TABLO
# ============================================================

def run_all(instances=INSTANCES, show_plots=True):
    rows = []
    for inst in instances:
        try:
            row = full_compare(inst, show_plot=show_plots)
            rows.append(row)
        except FileNotFoundError as e:
            print(f"!!! {inst}: dosya bulunamadi -> {e.filename}")
        except Exception as e:
            print(f"!!! {inst}: hata -> {e}")

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    """Iki metodu ozet metriklerle karsilastir."""
    metrics = ["diff_mean", "inside_mean", "outside_leak_ratio",
               "boundary_ratio", "hf_energy_ratio", "color_consistency",
               "ssim", "psnr", "lpips"]

    summary_rows = []
    for m in metrics:
        h_col, r_col = f"harmoni_{m}", f"rspaint_{m}"
        if h_col not in df.columns:
            continue
        summary_rows.append({
            "metric": m,
            "harmoni_mean": df[h_col].mean(),
            "rspaint_mean": df[r_col].mean(),
            "harmoni_wins": (df[h_col] < df[r_col]).sum() if m not in ("ssim", "psnr") else (df[h_col] > df[r_col]).sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== OZET (dusuk=iyi olan metrikler icin harmoni_wins sayisi anlamli) ===")
    print(summary_df.round(3).to_string(index=False))

    # mask alani ile hata iliskisi
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mask_area_px"], df["harmoni_diff_mean"], label="HarmoniDiff", alpha=0.7)
    ax.scatter(df["mask_area_px"], df["rspaint_diff_mean"], label="RSPaint", alpha=0.7)
    ax.set_xlabel("mask alani (piksel)")
    ax.set_ylabel("ortalama fark skoru")
    ax.legend()
    ax.set_title("Mask alani vs fark skoru")
    plt.tight_layout()
    plt.show()

    return summary_df


# ============================================================
# 10) CALISTIRMA ORNEGI
# ============================================================
df = run_all(INSTANCES, show_plots=True)
summary = summarize(df)
df.to_csv(f"{ROOT}/outputs/eval_summary.csv", index=False)