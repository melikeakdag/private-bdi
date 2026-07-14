"""
BRIGHT verisi icin sentetik damaged/destroyed bina uretim pipeline'i.

Mantik:
  - Her sahnede (mask tif) SAGLAM (intact=1) bina instance'lari cikarilir.
  - Her sahne icin crop SADECE AYNI SAHNEDEN (CSV'deki ayni image_name) secilir -
    baska sahnelerden/event'lerden crop ALINMAZ (radyometrik/sensor tutarliligi icin).
  - Bir sahnede birden fazla intact instance varsa, o sahnenin kendi crop'lari
    arasinda en iyi boyut (pixel_area) eslesmesini veren instance secilir (instance
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
        --crops_dir /content/drive/MyDrive/SAR/BRIGHT_Data/damagedbuildings \
        --mask_suffix _building_damage.tif \
        --bg_suffix .tif \
        --out_dir ./synthetic_pairs \
        --seed 42
"""
import argparse
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
                arr8 = np.stack([arr8] * 3, axis=-1)  # (H,W) -> (H,W,3), RGB icin sart
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


def process_scene(scene_id, masks_dir, images_dir, mask_suffix, bg_suffix,
                   scene_crops, min_area, image_size, out_dir):
    """scene_crops: SADECE bu scene_id'ye ait CSV satirlarinin listesi (aynı sahne kisitlamasi)."""
    mask_path = os.path.join(masks_dir, f"{scene_id}{mask_suffix}")
    bg_path = os.path.join(images_dir, f"{scene_id}{bg_suffix}")

    if not os.path.exists(mask_path) or not os.path.exists(bg_path):
        return None, "missing_file"

    if not scene_crops:
        return None, "pool_empty"

    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)
        mask_profile = src.profile

    instances = get_intact_instances(mask_arr, min_area)
    if not instances:
        return None, "no_intact_instances"

    # o sahnedeki her instance adayi icin, YINE O SAHNENIN kendi crop'lari arasinda
    # en yakin alanli olani bul (baska sahnelerden crop ALINMIYOR - ayni goruntu kisitlamasi)
    best = None  # (diff, instance, crop_rec)
    for inst in instances:
        for crop_rec in scene_crops:
            diff = abs(crop_rec["pixel_area"] - inst["area"])
            if best is None or diff < best[0]:
                best = (diff, inst, crop_rec)

    if best is None:
        return None, "pool_empty"

    _, inst, crop_rec = best

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
    parser.add_argument("--crops_dir", type=str, required=True,
                         help="Crop png dosyalarinin GERCEK bulundugu klasor. "
                              "CSV'deki saved_crop_path sutunundaki klasor eski/tasinmis olabilir; "
                              "sadece dosya adi (basename) alinip bu klasorde aranir.")
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

    # CSV'deki saved_crop_path eski/tasinmis klasor yolu tasiyor olabilir.
    # Sadece dosya adini (basename) alip GERCEK crops_dir icinde ariyoruz.
    df["saved_crop_path"] = df["saved_crop_path"].apply(
        lambda p: os.path.join(args.crops_dir, os.path.basename(str(p)))
    )
    n_before = len(df)
    exists_mask = df["saved_crop_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"UYARI: crops_dir icinde bulunamayan {n_missing}/{n_before} crop CSV'den cikarilacak "
              f"(muhtemelen sadece bir kismini kopyaladin, orn. sadece earthquake).")
        df = df[exists_mask].reset_index(drop=True)
    print(f"Kullanilabilir crop satiri: {len(df)}")

    # Sahneye gore grupla - HER SAHNE SADECE KENDI crop'larini gorebilir (ayni goruntu kisitlamasi)
    crops_by_scene = {
        scene_id: group.to_dict("records")
        for scene_id, group in df.groupby("image_name")
    }
    scene_ids = sorted(crops_by_scene.keys())
    random.shuffle(scene_ids)
    print(f"Islenecek sahne sayisi: {len(scene_ids)}")

    stats = {"ok": 0, "missing_file": 0, "no_intact_instances": 0, "pool_empty": 0, "error": 0}
    errors = []
    for scene_id in scene_ids:
        try:
            _, status = process_scene(
                scene_id, args.masks_dir, args.images_dir,
                args.mask_suffix, args.bg_suffix,
                crops_by_scene[scene_id], args.min_area, args.image_size, args.out_dir,
            )
            stats[status] += 1
        except Exception as e:
            stats["error"] += 1
            errors.append((scene_id, str(e)))
            print(f"HATA (atlaniyor) - {scene_id}: {e}")

    print("\n--- Ozet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print(f"\n{len(errors)} sahnede hata olustu (ilk 10):")
        for scene_id, msg in errors[:10]:
            print(f"  {scene_id}: {msg}")
    print(f"\nKullanilan crop: {stats['ok']} / Toplam mevcut crop: {len(df)} "
          f"(not: reuse yok ama sahne-ici crop'lardan sadece 1'i secildigi icin "
          f"kalan crop'lar o sahnenin diger, secilmeyen crop'lari)")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
