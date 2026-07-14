"""
Her sahne icin: sinif-maskesinden (0=bg,1=intact,2=damaged,3=destroyed) SAGLAM
(intact) bina instance'larini cikarir. Her instance icin:
  - bg.png    : tum sahne (background), image_size'a resize edilmis
  - mask.png  : o instance'in binary konum maskesi (255=yapistirilacak yer),
                bg ile ayni boyuta resize edilmis (NEAREST ile, kenar bozulmasin diye)
  - fg.png    : ayni sahneden (mumkunse) secilen damaged/destroyed crop,
                instance bbox boyutuna resize edilmis
  - meta.json : xy (normalize), bbox (piksel), instance_id, kaynak crop yolu

Boylece HarmoniDiff'in istedigi (background, mask, foreground) uclusu her
instance icin otomatik uretilmis olur.

Kullanim:
    python build_pasting_inputs.py \
        --mask_path /path/to/scene_mask.tif \
        --bg_path   /path/to/scene_background.tif \
        --crops_dir /path/to/damagedbuildings \
        --scene_id  haiti-earthquake_000 \
        --out_dir   ./pasting_inputs \
        --undamaged_value 1 \
        --same_scene_only
"""
import argparse
import glob
import json
import os

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage


def read_raster_as_pil(path):
    """tif/png farketmeksizin okuyup 8-bit RGB PIL donduren yardimci fonksiyon."""
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
    else:
        return Image.open(path).convert("RGB")


def get_undamaged_instances(mask_path, undamaged_value, min_area):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    binary = (mask == undamaged_value).astype(np.uint8)
    labeled, n = ndimage.label(binary)

    instances = []
    for inst_id in range(1, n + 1):
        ys, xs = np.where(labeled == inst_id)
        area = len(ys)
        if area < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        inst_mask = (labeled == inst_id).astype(np.uint8) * 255
        instances.append({
            "instance_id": inst_id,
            "bbox": (x1, y1, x2, y2),
            "area": area,
            "mask": inst_mask,  # tam sahne boyutunda binary mask
        })
    return instances, mask.shape


def pick_matching_crop(crop_paths, target_w, target_h):
    """Hedef bbox boyutuna en yakin en/boy oranina sahip crop'u secer."""
    target_ratio = target_w / max(target_h, 1)
    best_path, best_diff = None, float("inf")
    for p in crop_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        ratio = w / max(h, 1)
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff, best_path = diff, p
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--bg_path", type=str, required=True)
    parser.add_argument("--crops_dir", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True,
                         help="Sahne adi, crop dosyalarini filtrelemek icin (ornek: haiti-earthquake_000)")
    parser.add_argument("--out_dir", type=str, default="./pasting_inputs")
    parser.add_argument("--undamaged_value", type=int, default=1,
                         help="Mask'ta 'saglam/intact' sinifinin piksel degeri (BRIGHT varsayilani: 1)")
    parser.add_argument("--min_area", type=int, default=200,
                         help="Gurultu/cok kucuk parcalari elemek icin minimum piksel alani")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--same_scene_only", action="store_true",
                         help="Crop havuzunu sadece ayni scene_id ile baslayan dosyalarla sinirla")
    parser.add_argument("--max_instances", type=int, default=None,
                         help="Test icin islenecek instance sayisini sinirlar")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- crop havuzunu hazirla ---
    all_crops = sorted(glob.glob(os.path.join(args.crops_dir, "*")))
    all_crops = [p for p in all_crops if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
    if args.same_scene_only:
        pool = [p for p in all_crops if os.path.basename(p).startswith(args.scene_id)]
        if not pool:
            print(f"UYARI: '{args.scene_id}' ile baslayan crop bulunamadi, tum havuz kullanilacak.")
            pool = all_crops
    else:
        pool = all_crops

    if not pool:
        raise SystemExit(f"crops_dir icinde hic gecerli goruntu bulunamadi: {args.crops_dir}")
    print(f"Kullanilabilir crop sayisi: {len(pool)}")

    # --- sahne mask + background ---
    instances, mask_shape = get_undamaged_instances(args.mask_path, args.undamaged_value, args.min_area)
    print(f"Bulunan saglam (intact) bina instance sayisi: {len(instances)} (mask boyutu={mask_shape})")

    if not instances:
        raise SystemExit("Hic saglam bina instance'i bulunamadi - undamaged_value degerini kontrol et "
                          "(inspect_mask.py ile dogrula).")

    bg_full = read_raster_as_pil(args.bg_path)
    if bg_full.size != (mask_shape[1], mask_shape[0]):
        print(f"UYARI: bg boyutu {bg_full.size} != mask boyutu {(mask_shape[1], mask_shape[0])}. "
              f"Kayit/registration farki olabilir, bbox'lar kaymis olabilir.")

    if args.max_instances:
        instances = instances[: args.max_instances]

    written = 0
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        w, h = x2 - x1, y2 - y1

        crop_path = pick_matching_crop(pool, w, h)
        if crop_path is None:
            print(f"  instance {inst['instance_id']}: uygun crop bulunamadi, atlaniyor.")
            continue

        inst_dir = os.path.join(args.out_dir, f"{args.scene_id}_inst{inst['instance_id']:04d}")
        os.makedirs(inst_dir, exist_ok=True)

        # bg: tum sahne, image_size'a resize
        bg_resized = bg_full.resize((args.image_size, args.image_size))
        bg_resized.save(os.path.join(inst_dir, "bg.png"))

        # mask: instance'in binary konum maskesi, ayni sekilde resize (NEAREST!)
        mask_pil = Image.fromarray(inst["mask"], mode="L")
        mask_resized = mask_pil.resize((args.image_size, args.image_size), resample=Image.NEAREST)
        mask_resized.save(os.path.join(inst_dir, "mask.png"))

        # fg: secilen damaged/destroyed crop, bbox boyutuna resize
        fg_img = read_raster_as_pil(crop_path).resize((w, h))
        fg_img.save(os.path.join(inst_dir, "fg.png"))

        px, py = x1 / mask_shape[1], y1 / mask_shape[0]
        pw, ph = w / mask_shape[1], h / mask_shape[0]

        meta = {
            "scene_id": args.scene_id,
            "instance_id": inst["instance_id"],
            "bbox_px": [x1, y1, x2, y2],
            "xy_normalized": [px, py],
            "wh_normalized": [pw, ph],
            "area_px": inst["area"],
            "source_crop": crop_path,
        }
        with open(os.path.join(inst_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        written += 1

    print(f"\nTamamlandi: {written}/{len(instances)} instance icin (bg,mask,fg,meta) uretildi -> {args.out_dir}")


if __name__ == "__main__":
    main()
