"""
BRIGHT verisi icin sentetik damaged/destroyed bina uretim pipeline'i.

Mantik:
  - Her sahnede (mask tif) SAGLAM (intact=1) bina instance'lari cikarilir.
  - CSV'deki damaged/destroyed crop havuzundan, sahne basina SADECE 1 crop
    kullanilir, HIC TEKRAR YOK (reuse=1) -> crop bir kez kullanildiysa havuzdan cikar.
  - Bir sahnede birden fazla intact instance varsa, o sahne icin ULASILABILEN
    en iyi boyut (pixel_area) eslesmesini veren instance secilir (instance
    secimi + crop eslestirme BIRLIKTE optimize edilir).
  - Cikti: bg.png / mask_location.png (binary yapistirma yeri) / fg.png /
    updated_label.tif (orijinal cok-sinifli mask'in GUNCELLENMIS hali - secilen
    instance'in pikselleri artik crop'un gercek damage_class'ina esitlenir) /
    meta.json

CSV semasi (kullanici tarafindan onaylandi):
    image_name, building_id, damage_class, class_name,
    row_min, row_max, col_min, col_max, centroid_row, centroid_col,
    pixel_area, saved_crop_path

Kullanim:
    python build_synthetic_pairs.py \
        --csv_path damaged_buildings.csv \
        --masks_dir /content/drive/MyDrive/SAR/BRIGHT_Data/target \
        --images_dir /content/drive/MyDrive/SAR/BRIGHT_Data/images \
        --mask_suffix _building_damage.tif \
        --bg_suffix .tif \
        --out_dir ./synthetic_pairs \
        --seed 42
"""
import argparse
import bisect
import json
import os
import random

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy import ndimage

INTACT_VALUE = 1


def read_raster_as_pil(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            n_bands = src.count
            if n_bands >= 3:
                arr = src.read([1, 2, 3]).astype(float)
                arr = np.transpose(arr, (1, 2, 0))
            else:
                band = src.read(1).astype(float)
                p2, p98 = np.percentile(band, (2, 98))
                stretched = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
                arr8 = (stretched * 255).astype(np.uint8)
                return Image.fromarray(arr8, mode="RGB")
            p2 = np.percentile(arr, 2, axis=(0, 1))
            p98 = np.percentile(arr, 98, axis=(0, 1))
            stretched = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
            arr8 = (stretched * 255).astype(np.uint8)
            return Image.fromarray(arr8, mode="RGB")
    return Image.open(path).convert("RGB")


def get_intact_instances(mask_arr, min_area):
    binary = (mask_arr == INTACT_VALUE).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    instances = []
    for inst_id in range(1, n + 1):
        ys, xs = np.where(labeled == inst_id)
        area = len(ys)
        if area < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        instances.append({
            "instance_id": inst_id,
            "bbox": (x1, y1, x2, y2),
            "area": area,
            "mask": (labeled == inst_id),
        })
    return instances


class CropPool:
    """Alan (pixel_area) bazli, tekrar-kullanimsiz (reuse=1) en-yakin-komsu havuzu."""

    def __init__(self, crops_df):
        recs = crops_df.to_dict("records")
        recs.sort(key=lambda r: r["pixel_area"])
        self.areas = [r["pixel_area"] for r in recs]
        self.refs = recs

    def peek_nearest(self, target_area):
        """Havuzdan cikarmadan en yakin alanli crop'un index'ini dondurur."""
        if not self.areas:
            return None
        pos = bisect.bisect_left(self.areas, target_area)
        candidates = []
        if pos < len(self.areas):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best_idx = min(candidates, key=lambda i: abs(self.areas[i] - target_area))
        return best_idx

    def pop(self, idx):
        rec = self.refs.pop(idx)
        self.areas.pop(idx)
        return rec

    def __len__(self):
        return len(self.areas)


def process_scene(scene_id, masks_dir, images_dir, mask_suffix, bg_suffix,
                   pool, min_area, image_size, out_dir):
    mask_path = os.path.join(masks_dir, f"{scene_id}{mask_suffix}")
    bg_path = os.path.join(images_dir, f"{scene_id}{bg_suffix}")

    if not os.path.exists(mask_path) or not os.path.exists(bg_path):
        return None, "missing_file"

    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)
        mask_profile = src.profile

    instances = get_intact_instances(mask_arr, min_area)
    if not instances:
        return None, "no_intact_instances"

    # o sahnedeki her instance adayi icin en yakin crop'u bul (havuzdan cikarmadan)
    best = None  # (diff, instance, pool_idx)
    for inst in instances:
        idx = pool.peek_nearest(inst["area"])
        if idx is None:
            continue
        diff = abs(pool.areas[idx] - inst["area"])
        if best is None or diff < best[0]:
            best = (diff, inst, idx)

    if best is None:
        return None, "pool_empty"

    _, inst, pool_idx = best
    crop_rec = pool.pop(pool_idx)  # artik bu crop bir daha kullanilamaz

    x1, y1, x2, y2 = inst["bbox"]
    w, h = x2 - x1, y2 - y1
    damage_class = int(crop_rec["damage_class"])

    scene_out_dir = os.path.join(out_dir, f"{scene_id}_inst{inst['instance_id']:04d}")
    os.makedirs(scene_out_dir, exist_ok=True)

    # bg
    bg_full = read_raster_as_pil(bg_path)
    bg_full.resize((image_size, image_size)).save(os.path.join(scene_out_dir, "bg.png"))

    # mask_location (binary, sadece bu instance)
    loc_mask = (inst["mask"].astype(np.uint8) * 255)
    Image.fromarray(loc_mask, mode="L").resize(
        (image_size, image_size), resample=Image.NEAREST
    ).save(os.path.join(scene_out_dir, "mask_location.png"))

    # fg
    fg_img = read_raster_as_pil(crop_rec["saved_crop_path"]).resize((w, h))
    fg_img.save(os.path.join(scene_out_dir, "fg.png"))

    # updated_label: orijinal mask'in kopyasi, instance pikselleri damage_class'a guncellenmis
    updated = mask_arr.copy()
    updated[inst["mask"]] = damage_class
    upd_profile = mask_profile.copy()
    upd_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(os.path.join(scene_out_dir, "updated_label.tif"), "w", **upd_profile) as dst:
        dst.write(updated.astype(np.uint8), 1)

    H, W = mask_arr.shape
    meta = {
        "scene_id": scene_id,
        "instance_id": inst["instance_id"],
        "bbox_px": [x1, y1, x2, y2],
        "xy_normalized": [x1 / W, y1 / H],
        "wh_normalized": [w / W, h / H],
        "instance_area_px": inst["area"],
        "matched_crop_area_px": int(crop_rec["pixel_area"]),
        "area_diff_px": int(abs(crop_rec["pixel_area"] - inst["area"])),
        "damage_class": damage_class,
        "class_name": crop_rec["class_name"],
        "source_crop": crop_rec["saved_crop_path"],
    }
    with open(os.path.join(scene_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_building_damage.tif")
    parser.add_argument("--bg_suffix", type=str, default=".tif")
    parser.add_argument("--out_dir", type=str, default="./synthetic_pairs")
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required_cols = {"image_name", "damage_class", "class_name", "pixel_area", "saved_crop_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV'de eksik sutun(lar): {missing}")

    pool = CropPool(df)
    print(f"Baslangic crop havuzu: {len(pool)}")

    scene_ids = sorted(df["image_name"].unique().tolist())
    random.shuffle(scene_ids)
    print(f"Islenecek sahne sayisi: {len(scene_ids)}")

    stats = {"ok": 0, "missing_file": 0, "no_intact_instances": 0, "pool_empty": 0}
    for scene_id in scene_ids:
        _, status = process_scene(
            scene_id, args.masks_dir, args.images_dir,
            args.mask_suffix, args.bg_suffix,
            pool, args.min_area, args.image_size, args.out_dir,
        )
        stats[status] += 1

    print("\n--- Ozet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Kalan kullanilmamis crop: {len(pool)}")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
