"""
merge_datasets.py
==================
synthetic_pairs/ + HarmoniDiff sonuclari + RS-Paint sonuclarini, BRIGHT'in
kendi native dizin mimarisine (pre-event/ post-event/ target/, duz klasor +
eslesen dosya adlari) uygun sekilde birlestirir. Sadece change detection
modelinin gercekten okuyacagi dosyalar kopyalanir - fg.png, mask_location.png,
clip_similarity.txt, instance-basina meta.json GIBI uretim ara urunleri
DAHIL EDILMEZ (bu bilgiler zaten manifest.csv'de ozetleniyor).

Cikti yapisi:
    merged_dataset/
    |-- pre-event/
    |   `-- {instance_id}_pre_disaster.png       <- synthetic_pairs/.../bg.png (pre-disaster SAR, degismeyen sahne)
    |-- post-event_harmonidiff/
    |   `-- {instance_id}_post_disaster.png      <- harmonidiff_dir/.../result.png
    |-- post-event_rspaint/
    |   `-- {instance_id}_post_disaster.png      <- rspaint_dir/.../result_raw.png
    |-- target/
    |   `-- {instance_id}_building_damage.tif    <- synthetic_pairs/.../updated_label.tif (TEK, ikisi de paylasir)
    `-- manifest.csv

Kullanim:
    python merge_datasets.py \
        --synthetic_dir   /content/drive/MyDrive/SynDataGen/outputs/synthetic_pairs \
        --harmonidiff_dir /content/drive/MyDrive/SynDataGen/outputs/harmonized \
        --rspaint_dir     /content/drive/MyDrive/SynDataGen/outputs/rspaint_results \
        --out_dir         /content/drive/MyDrive/SynDataGen/outputs/merged_dataset
"""
import argparse
import csv
import json
import os
import re
import shutil


def extract_scene_and_event(instance_id):
    """'turkey-earthquake_00000602_inst0002' -> scene='turkey-earthquake_00000602', event='turkey-earthquake'"""
    m = re.match(r"^(.*)_inst\d+$", instance_id)
    scene_id = m.group(1) if m else instance_id
    event = re.sub(r"_\d+$", "", scene_id)
    return scene_id, event


def find_harmonidiff_result(harmonidiff_dir, instance_id):
    if not harmonidiff_dir:
        return None
    path = os.path.join(harmonidiff_dir, instance_id, "result.png")
    return path if os.path.exists(path) else None


def find_rspaint_result(rspaint_dir, instance_id):
    """{inst}/result.png - mask-disi orijinal bg'ye geri dondurulmus, NIHAI/post-process
    edilmis hali (HarmoniDiff'teki result.png ile ayni disiplin, adil kiyaslama icin sart)."""
    if not rspaint_dir:
        return None
    path = os.path.join(rspaint_dir, instance_id, "result.png")
    return path if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--harmonidiff_dir", type=str, default=None)
    parser.add_argument("--rspaint_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--require_both", action="store_true", default=True,
                         help="Sadece HER IKI modelin de sonucu olan instance'lari dahil et (varsayilan)")
    parser.add_argument("--no_require_both", dest="require_both", action="store_false")
    args = parser.parse_args()

    if not args.harmonidiff_dir and not args.rspaint_dir:
        raise SystemExit("En az bir tanesini vermelisin: --harmonidiff_dir ve/veya --rspaint_dir")

    pre_dir = os.path.join(args.out_dir, "pre-event")
    post_hd_dir = os.path.join(args.out_dir, "post-event_harmonidiff")
    post_rs_dir = os.path.join(args.out_dir, "post-event_rspaint")
    target_dir = os.path.join(args.out_dir, "target")
    for d in (pre_dir, post_hd_dir, post_rs_dir, target_dir):
        os.makedirs(d, exist_ok=True)

    all_ids = sorted(
        d for d in os.listdir(args.synthetic_dir)
        if os.path.isdir(os.path.join(args.synthetic_dir, d))
    )
    print(f"synthetic_pairs altinda bulunan instance: {len(all_ids)}")

    manifest_rows = []
    stats = {"merged": 0, "skipped_missing_source": 0, "skipped_missing_model": 0}

    for inst_id in all_ids:
        src_dir = os.path.join(args.synthetic_dir, inst_id)
        pre_sar_path = os.path.join(src_dir, "bg.png")  # bg.png = pre-disaster SAR goruntusu (hasar eklenmemis, degismeyen hal)
        meta_path = os.path.join(src_dir, "meta.json")
        label_path = os.path.join(src_dir, "updated_label.tif")

        if not (os.path.exists(pre_sar_path) and os.path.exists(meta_path) and os.path.exists(label_path)):
            stats["skipped_missing_source"] += 1
            continue

        harmonidiff_result = find_harmonidiff_result(args.harmonidiff_dir, inst_id)
        rspaint_result = find_rspaint_result(args.rspaint_dir, inst_id)
        has_harmonidiff = harmonidiff_result is not None
        has_rspaint = rspaint_result is not None

        if args.harmonidiff_dir and args.rspaint_dir and args.require_both:
            model_requirement_met = has_harmonidiff and has_rspaint
        elif args.harmonidiff_dir and not args.rspaint_dir:
            model_requirement_met = has_harmonidiff
        elif args.rspaint_dir and not args.harmonidiff_dir:
            model_requirement_met = has_rspaint
        else:
            model_requirement_met = has_harmonidiff or has_rspaint

        if not model_requirement_met:
            stats["skipped_missing_model"] += 1
            continue

        # pre + target (ORTAK, iki model de paylasir)
        shutil.copy(pre_sar_path, os.path.join(pre_dir, f"{inst_id}_pre_disaster.png"))
        shutil.copy(label_path, os.path.join(target_dir, f"{inst_id}_building_damage.tif"))

        # post (model-basina ayri klasor)
        if has_harmonidiff:
            shutil.copy(harmonidiff_result, os.path.join(post_hd_dir, f"{inst_id}_post_disaster.png"))
        if has_rspaint:
            shutil.copy(rspaint_result, os.path.join(post_rs_dir, f"{inst_id}_post_disaster.png"))

        with open(meta_path) as f:
            meta = json.load(f)
        scene_id, event = extract_scene_and_event(inst_id)

        manifest_rows.append({
            "instance_id": inst_id,
            "scene_id": scene_id,
            "event": event,
            "damage_class": meta.get("damage_class", ""),
            "class_name": meta.get("class_name", ""),
            "instance_area_px": meta.get("instance_area_px", meta.get("area", "")),
            "has_harmonidiff": has_harmonidiff,
            "has_rspaint": has_rspaint,
        })
        stats["merged"] += 1

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    fieldnames = ["instance_id", "scene_id", "event", "damage_class", "class_name",
                  "instance_area_px", "has_harmonidiff", "has_rspaint"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n--- Ozet ---")
    print(f"Birlestirilen instance:               {stats['merged']}")
    print(f"Atlanan (pre/label/meta eksik):        {stats['skipped_missing_source']}")
    print(f"Atlanan (istenen model sonucu eksik):   {stats['skipped_missing_model']}")
    print(f"Manifest: {manifest_path}")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
