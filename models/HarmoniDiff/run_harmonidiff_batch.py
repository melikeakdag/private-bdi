"""
outputs/synthetic_pairs/ altindaki TUM instance klasorlerini gezip HarmoniDiff ile
harmonize eder. run_harmonidiff_single.py ile ayni mantik, ama:
  - Tum klasorleri otomatik bulur (tek tek isim vermene gerek yok)
  - Zaten islenmis (result.png var olan) instance'lari ATLAR -> resume edilebilir
    (Colab baglantisi koparsa, tekrar calistirinca kaldigi yerden devam eder)
  - Her N instance'ta bir ilerleme durumu yazdirir
  - Tek instance'ta hata olursa atlar, batch durmaz, sonda ozet basar

models/HarmoniDiff/ klasorunun ICINDE durmali (relative import).

Kullanim:
    cd models/HarmoniDiff
    python run_harmonidiff_batch.py \
        --synthetic_dir /content/drive/MyDrive/SAR/SynDataGen/outputs/synthetic_pairs \
        --out_dir /content/drive/MyDrive/SAR/SynDataGen/outputs/harmonized \
        --limit 0   # 0 = hepsi, test icin orn. 10 verebilirsin
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from pipeline_harmonidiff import HarmoniDiffPipeline
from discrimination import ResNet4ch

REQUIRED_META_FIELDS = ["bg_prompt", "fg_prompt", "longitude", "latitude",
                         "bg_gsd", "cloud_cover", "year", "month", "day"]


def load_instance(inst_dir):
    bg = Image.open(os.path.join(inst_dir, "bg.png")).convert("RGB")
    mask = Image.open(os.path.join(inst_dir, "mask_location.png")).convert("L")
    fg = Image.open(os.path.join(inst_dir, "fg.png")).convert("RGB")
    with open(os.path.join(inst_dir, "meta.json")) as f:
        meta = json.load(f)
    return bg, mask, fg, meta


def harmonize_one(pipe, inst_dir, out_dir, image_size, num_inference_steps, edge_width_ratio):
    bg, mask, fg, meta = load_instance(inst_dir)

    missing = [k for k in REQUIRED_META_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"meta.json eksik alan(lar): {missing} - once enrich_meta.py calistir")

    metadata = [meta["longitude"], meta["latitude"], meta["bg_gsd"], meta["cloud_cover"],
                meta["year"], meta["month"], meta["day"]]
    px, py = meta["xy_normalized"]

    bg_resized = bg.resize((image_size, image_size))
    fg_w = max(1, round(meta["wh_normalized"][0] * image_size))
    fg_h = max(1, round(meta["wh_normalized"][1] * image_size))
    fg_resized = fg.resize((fg_w, fg_h))

    result = pipe(
        bg_prompt=meta["bg_prompt"], fg_prompt=meta["fg_prompt"],
        bg_image=bg_resized, fg_image=fg_resized,
        xy=[px, py], metadata=metadata,
        num_inference_steps=num_inference_steps,
        height=image_size, width=image_size,
        edge_width_ratio=edge_width_ratio,
        viz_checkpoints=None,
    )

    mask_resized = mask.resize((image_size, image_size), resample=Image.NEAREST)
    mask_arr = (np.array(mask_resized) > 127).astype(np.float32)[..., None]
    result_arr = np.array(result.convert("RGB")).astype(np.float32)
    bg_arr = np.array(bg_resized).astype(np.float32)
    final_arr = (mask_arr * result_arr + (1 - mask_arr) * bg_arr).astype(np.uint8)
    final_img = Image.fromarray(final_arr)

    inst_name = os.path.basename(inst_dir.rstrip("/"))
    inst_out_dir = os.path.join(out_dir, inst_name)
    os.makedirs(inst_out_dir, exist_ok=True)

    result.save(os.path.join(inst_out_dir, "result_raw.png"))
    final_img.save(os.path.join(inst_out_dir, "result.png"))
    with open(os.path.join(inst_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return inst_out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="BiliSakura/DiffusionSat-Single-512")
    parser.add_argument("--discriminator_path", type=str, default="./checkpoints/discriminator.pt")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--edge_width_ratio", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--limit", type=int, default=0, help="0 = hepsi, test icin kucuk bir sayi verebilirsin")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_instances = sorted([
        d for d in os.listdir(args.synthetic_dir)
        if os.path.isdir(os.path.join(args.synthetic_dir, d))
    ])
    if args.limit > 0:
        all_instances = all_instances[:args.limit]
    print(f"Toplam instance: {len(all_instances)}")

    # resume: zaten result.png'si olanlari atla
    todo = []
    already_done = 0
    for inst_name in all_instances:
        result_path = os.path.join(args.out_dir, inst_name, "result.png")
        if os.path.exists(result_path):
            already_done += 1
        else:
            todo.append(inst_name)
    print(f"Zaten islenmis (atlanacak): {already_done}")
    print(f"Islenecek: {len(todo)}")

    if not todo:
        print("Yapilacak bir sey yok, hepsi zaten islenmis.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    print(f"Model yukleniyor ({args.model_id}, device={device})...")
    discriminator = ResNet4ch.from_pretrained(args.discriminator_path, device=device)
    pipe = HarmoniDiffPipeline.from_pretrained_custom(
        args.model_id, discriminator=discriminator, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype]
    ).to(device)
    print("Model hazir. Isleme basliyor...\n")

    stats = {"ok": 0, "error": 0}
    errors = []
    start_time = time.time()

    for i, inst_name in enumerate(todo, 1):
        inst_dir = os.path.join(args.synthetic_dir, inst_name)
        try:
            harmonize_one(pipe, inst_dir, args.out_dir, args.image_size,
                          args.num_inference_steps, args.edge_width_ratio)
            stats["ok"] += 1
        except Exception as e:
            stats["error"] += 1
            errors.append((inst_name, str(e)))
            print(f"HATA - {inst_name}: {e}")

        if i % args.log_every == 0 or i == len(todo):
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = avg * (len(todo) - i)
            print(f"[{i}/{len(todo)}] ok={stats['ok']} error={stats['error']} "
                  f"| gecen={elapsed/60:.1f}dk | kalan_tahmini={remaining/60:.1f}dk")

    print("\n--- Ozet ---")
    print(f"Bu calistirmada islenen: {len(todo)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print(f"\nHata alan instance'lar (ilk 10):")
        for inst_name, msg in errors[:10]:
            print(f"  {inst_name}: {msg}")
    print(f"\nToplam tamamlanan (onceki + bu calistirma): {already_done + stats['ok']} / {len(all_instances)}")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
