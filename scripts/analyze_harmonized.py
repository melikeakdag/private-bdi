"""
outputs/synthetic_pairs/ + outputs/harmonized/ klasorlerini birlikte tarayip
her instance icin mask alani, harmonizasyon "fark" istatistikleri, event/damage_class
bilgisini birlestirip tek bir CSV + ozet rapor uretir.

Bagimsiz calistirilabilir (model yuklemez, sadece PIL/numpy/pandas kullanir,
HarmoniDiff repo'suna import bagimliligi YOK - istedigin yerden calistirabilirsin).

Kullanim (notebook icinde):
    %run analyze_harmonized.py --synthetic_dir {ROOT}/outputs/synthetic_pairs \
                                --harmonized_dir {ROOT}/outputs/harmonized \
                                --out_csv {ROOT}/outputs/eda_report.csv
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image


def extract_event(scene_id):
    return re.sub(r'_\d+$', '', scene_id)


def analyze_instance(inst_name, synthetic_dir, harmonized_dir):
    syn_dir = os.path.join(synthetic_dir, inst_name)
    harm_dir = os.path.join(harmonized_dir, inst_name)

    meta_path = os.path.join(syn_dir, "meta.json")
    mask_path = os.path.join(syn_dir, "mask_location.png")
    bg_path = os.path.join(syn_dir, "bg.png")
    result_path = os.path.join(harm_dir, "result.png")
    result_raw_path = os.path.join(harm_dir, "result_raw.png")

    if not (os.path.exists(meta_path) and os.path.exists(mask_path)
            and os.path.exists(bg_path) and os.path.exists(result_path)):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    mask = np.array(Image.open(mask_path).convert("L"))
    mask_bin = (mask > 127)
    mask_area_px = int(mask_bin.sum())
    mask_ratio = mask_area_px / mask_bin.size

    bg = np.array(Image.open(bg_path).convert("RGB")).astype(float)
    result = np.array(Image.open(result_path).convert("RGB").resize(bg.shape[1::-1])).astype(float)

    diff = np.abs(bg - result).sum(axis=-1)  # (H,W)

    # SADECE mask icindeki fark (asil ilgilendigimiz bolge)
    if mask_bin.shape != diff.shape:
        mask_bin_r = np.array(Image.fromarray(mask_bin.astype(np.uint8) * 255).resize(
            diff.shape[::-1], Image.NEAREST)) > 127
    else:
        mask_bin_r = mask_bin

    diff_in_mask = diff[mask_bin_r] if mask_bin_r.any() else np.array([0.0])
    diff_out_mask = diff[~mask_bin_r] if (~mask_bin_r).any() else np.array([0.0])

    row = {
        "instance": inst_name,
        "scene_id": meta.get("scene_id"),
        "event": extract_event(meta.get("scene_id", inst_name)),
        "damage_class": meta.get("damage_class"),
        "class_name": meta.get("class_name"),
        "instance_area_px": meta.get("instance_area_px"),
        "matched_crop_area_px": meta.get("matched_crop_area_px"),
        "area_diff_px": meta.get("area_diff_px"),
        "mask_area_px_512": mask_area_px,
        "mask_ratio_of_canvas": mask_ratio,
        "diff_in_mask_mean": float(diff_in_mask.mean()),
        "diff_in_mask_max": float(diff_in_mask.max()),
        "diff_out_mask_mean": float(diff_out_mask.mean()),  # ideal: ~0 (post-process disariyi degistirmemeli)
        "changed_px_ratio_in_mask": float((diff_in_mask > 10).mean()),
        "has_result_raw": os.path.exists(result_raw_path),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--harmonized_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    instances = sorted([
        d for d in os.listdir(args.harmonized_dir)
        if os.path.isdir(os.path.join(args.harmonized_dir, d))
    ])
    print(f"harmonized/ icinde bulunan instance sayisi: {len(instances)}")

    rows = []
    for inst_name in instances:
        row = analyze_instance(inst_name, args.synthetic_dir, args.harmonized_dir)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Analiz edilebilen instance: {len(df)}")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"CSV kaydedildi: {args.out_csv}")

    if len(df) == 0:
        print("Analiz edilecek veri yok, cikiliyor.")
        return

    print("\n=== Genel ozet ===")
    print(df[["mask_area_px_512", "diff_in_mask_mean", "diff_out_mask_mean",
               "changed_px_ratio_in_mask"]].describe())

    print("\n=== Event dagilimi ===")
    print(df["event"].value_counts())

    print("\n=== Damage class dagilimi ===")
    print(df["class_name"].value_counts())

    print("\n=== Event basina ortalama harmonizasyon farki (mask icinde) ===")
    print(df.groupby("event")["diff_in_mask_mean"].mean().sort_values(ascending=False))

    print("\n=== ⚠️ Post-process disari tasma kontrolu (diff_out_mask_mean idealde ~0) ===")
    leak_suspects = df.sort_values("diff_out_mask_mean", ascending=False).head(5)
    print(leak_suspects[["instance", "diff_out_mask_mean"]])

    print("\n=== En BUYUK mask alanina sahip 5 instance ===")
    print(df.sort_values("mask_area_px_512", ascending=False)
          [["instance", "mask_area_px_512", "class_name", "event"]].head(5))

    print("\n=== En KUCUK mask alanina sahip 5 instance ===")
    print(df.sort_values("mask_area_px_512", ascending=True)
          [["instance", "mask_area_px_512", "class_name", "event"]].head(5))

    print("\n=== Harmonizasyon farki EN DUSUK (supheli - hicbir sey degismemis olabilir) 5 instance ===")
    print(df.sort_values("diff_in_mask_mean", ascending=True)
          [["instance", "diff_in_mask_mean", "mask_area_px_512"]].head(5))

    print("\n=== Harmonizasyon farki EN YUKSEK (supheli - asiri degisim/artefakt olabilir) 5 instance ===")
    print(df.sort_values("diff_in_mask_mean", ascending=False)
          [["instance", "diff_in_mask_mean", "mask_area_px_512"]].head(5))


if __name__ == "__main__":
    main()
