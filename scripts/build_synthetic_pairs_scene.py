"""
BRIGHT verisi icin sentetik damaged/destroyed bina uretim pipeline'i (v3 - tek
klasor / sahne yapisi).

Onceki versiyondan (build_synthetic_pairs_k.py) farki: artik her sahne icin
5 AYRI instance klasoru degil, TEK bir sahne klasoru uretiliyor:

    outputs/synthetic_pairs_k5/{scene_id}/
        bg.png                 <- orijinal sahne (1 kez)
        updated_label.tif      <- secilen TUM instance'larin degisikligini iceren tek mask
        meta.json               <- sahne ozeti (kac instance, siniflar, vb.)
        instances/
            inst0004/
                mask_location.png
                fg.png
                meta.json       <- bu instance'a ozel detay
            inst0012/
                ...

Secim mantigi (ayni-sahne kisitlamasi + K>1) onceki versiyonla ayni:
sahne basina en fazla K instance, sadece o sahnenin KENDI crop'lariyla,
greedy en-iyi-boyut-eslestirme ile secilir.

Kullanim:
    python build_synthetic_pairs_scene.py \
        --csv_path data/damagedbuildings/damaged_buildings.csv \
        --masks_dir data/raw/target \
        --images_dir data/raw/images \
        --crops_dir data/damagedbuildings \
        --mask_suffix _building_damage.tif \
        --bg_suffix _post_disaster.tif \
        --out_dir outputs/synthetic_pairs_k5 \
        --k 5 \
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
                arr8 = np.stack([arr8] * 3, axis=-1)
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


def greedy_match_k(instances, crops, k):
    remaining_instances = list(instances)
    remaining_crops = list(crops)
    selected = []

    for _ in range(k):
        if not remaining_instances or not remaining_crops:
            break
        best = None
        for i, inst in enumerate(remaining_instances):
            for j, crop in enumerate(remaining_crops):
                diff = abs(crop["pixel_area"] - inst["area"])
                if best is None or diff < best[0]:
                    best = (diff, i, j)
        _, i, j = best
        inst = remaining_instances.pop(i)
        crop = remaining_crops.pop(j)
        selected.append((inst, crop))

    return selected


def process_scene(scene_id, masks_dir, images_dir, mask_suffix, bg_suffix,
                   scene_crops, min_area, image_size, out_dir, k):
    mask_path = os.path.join(masks_dir, f"{scene_id}{mask_suffix}")
    bg_path = os.path.join(images_dir, f"{scene_id}{bg_suffix}")

    if not os.path.exists(mask_path) or not os.path.exists(bg_path):
        return 0, "missing_file"
    if not scene_crops:
        return 0, "pool_empty"

    with rasterio.open(mask_path) as src:
        mask_arr = src.read(1)
        mask_profile = src.profile

    instances = get_intact_instances(mask_arr, min_area)
    if not instances:
        return 0, "no_intact_instances"

    selected = greedy_match_k(instances, scene_crops, k)
    if not selected:
        return 0, "pool_empty"

    # --- SAHNE klasoru (tek) ---
    scene_dir = os.path.join(out_dir, scene_id)
    instances_dir = os.path.join(scene_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    bg_full = read_raster_as_pil(bg_path)
    bg_resized = bg_full.resize((image_size, image_size))
    bg_resized.save(os.path.join(scene_dir, "bg.png"))

    # sahne seviyesinde TEK updated_label - tum secilenler birlikte
    updated = mask_arr.copy()
    for inst, crop in selected:
        updated[inst["mask"]] = int(crop["damage_class"])
    upd_profile = mask_profile.copy()
    upd_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(os.path.join(scene_dir, "updated_label.tif"), "w", **upd_profile) as dst:
        dst.write(updated.astype(np.uint8), 1)

    H, W = mask_arr.shape
    instance_summaries = []

    for inst, crop in selected:
        x1, y1, x2, y2 = inst["bbox"]
        w, h = x2 - x1, y2 - y1
        damage_class = int(crop["damage_class"])
        inst_name = f"inst{inst['instance_id']:04d}"

        inst_dir = os.path.join(instances_dir, inst_name)
        os.makedirs(inst_dir, exist_ok=True)

        loc_mask = (inst["mask"].astype(np.uint8) * 255)
        Image.fromarray(loc_mask, mode="L").resize(
            (image_size, image_size), resample=Image.NEAREST
        ).save(os.path.join(inst_dir, "mask_location.png"))

        fg_img = read_raster_as_pil(crop["saved_crop_path"]).resize((w, h))
        fg_img.save(os.path.join(inst_dir, "fg.png"))

        inst_meta = {
            "scene_id": scene_id,
            "instance_id": inst["instance_id"],
            "bbox_px": [x1, y1, x2, y2],
            "xy_normalized": [x1 / W, y1 / H],
            "wh_normalized": [w / W, h / H],
            "instance_area_px": inst["area"],
            "matched_crop_area_px": int(crop["pixel_area"]),
            "area_diff_px": int(abs(crop["pixel_area"] - inst["area"])),
            "damage_class": damage_class,
            "class_name": crop["class_name"],
            "source_crop": crop["saved_crop_path"],
        }
        with open(os.path.join(inst_dir, "meta.json"), "w") as f:
            json.dump(inst_meta, f, indent=2)

        instance_summaries.append({"instance": inst_name, "damage_class": damage_class,
                                    "class_name": crop["class_name"]})

    scene_meta = {
        "scene_id": scene_id,
        "k_requested": k,
        "k_selected": len(selected),
        "instances": instance_summaries,
    }
    with open(os.path.join(scene_dir, "meta.json"), "w") as f:
        json.dump(scene_meta, f, indent=2)

    return len(selected), "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--crops_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_building_damage.tif")
    parser.add_argument("--bg_suffix", type=str, default="_post_disaster.tif")
    parser.add_argument("--out_dir", type=str, default="./synthetic_pairs_scene")
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_scenes", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required_cols = {"image_name", "damage_class", "class_name", "pixel_area", "saved_crop_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV'de eksik sutun(lar): {missing}")

    df["saved_crop_path"] = df["saved_crop_path"].apply(
        lambda p: os.path.join(args.crops_dir, os.path.basename(str(p)))
    )
    exists_mask = df["saved_crop_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"UYARI: crops_dir icinde bulunamayan {n_missing} crop CSV'den cikarilacak.")
        df = df[exists_mask].reset_index(drop=True)
    print(f"Kullanilabilir crop satiri: {len(df)}")

    crops_by_scene = {
        scene_id: group.to_dict("records")
        for scene_id, group in df.groupby("image_name")
    }
    scene_ids = sorted(crops_by_scene.keys())
    random.shuffle(scene_ids)
    if args.limit_scenes > 0:
        scene_ids = scene_ids[:args.limit_scenes]
        print(f"UYARI: test modu - sadece {args.limit_scenes} sahne islenecek")
    print(f"Islenecek sahne sayisi: {len(scene_ids)}")
    print(f"Sahne basina hedef K: {args.k}")

    stats = {"ok": 0, "missing_file": 0, "no_intact_instances": 0, "pool_empty": 0, "error": 0}
    total_instances = 0
    errors = []

    for scene_id in scene_ids:
        try:
            n_written, status = process_scene(
                scene_id, args.masks_dir, args.images_dir,
                args.mask_suffix, args.bg_suffix,
                crops_by_scene[scene_id], args.min_area, args.image_size, args.out_dir, args.k,
            )
            stats[status] += 1
            total_instances += n_written
        except Exception as e:
            stats["error"] += 1
            errors.append((scene_id, str(e)))
            print(f"HATA (atlaniyor) - {scene_id}: {e}")

    print("\n--- Ozet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print(f"\n{len(errors)} sahnede hata (ilk 10):")
        for scene_id, msg in errors[:10]:
            print(f"  {scene_id}: {msg}")
    print(f"\nToplam sahne klasoru (ok): {stats['ok']}")
    print(f"Toplam instance (tum sahneler): {total_instances}")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
