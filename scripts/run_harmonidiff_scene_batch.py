"""
outputs/synthetic_pairs_k5/{scene_id}/ altindaki HER instance'i BAGIMSIZ olarak
(orijinal bg uzerinde) HarmoniDiff ile harmonize eder, sonra hepsini TEK bir
result.png'de birlestirir (mask'lar ortusmedigi icin sorunsuz birlesir).

Cikti: outputs/harmonized_k5/{scene_id}/result.png  (TEK foto, K hasarin hepsi)

models/HarmoniDiff/ klasorunun ICINDE durmali (relative import).

Kullanim:
    cd models/HarmoniDiff
    python run_harmonidiff_scene_batch.py \
        --scene_dir /path/to/outputs/synthetic_pairs_k5 \
        --out_dir /path/to/outputs/harmonized_k5 \
        --limit 0
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


def harmonize_instance(pipe, bg_resized, inst_dir, image_size, num_inference_steps, edge_width_ratio):
    """Tek bir instance'i, VERILEN (orijinal) bg uzerinde harmonize eder.
    O instance'in kendi maskesini ve harmonize edilmis goruntuyu dondurur."""
    mask = Image.open(os.path.join(inst_dir, "mask_location.png")).convert("L")
    fg = Image.open(os.path.join(inst_dir, "fg.png")).convert("RGB")
    with open(os.path.join(inst_dir, "meta.json")) as f:
        meta = json.load(f)

    missing = [k for k in REQUIRED_META_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"meta.json eksik alan(lar): {missing} - once enrich_meta.py calistir")

    metadata = [meta["longitude"], meta["latitude"], meta["bg_gsd"], meta["cloud_cover"],
                meta["year"], meta["month"], meta["day"]]
    px, py = meta["xy_normalized"]

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
    mask_arr = (np.array(mask_resized) > 127).astype(np.float32)
    result_arr = np.array(result.convert("RGB")).astype(np.float32)

    return mask_arr, result_arr


def process_scene(pipe, scene_dir, out_dir, image_size, num_inference_steps, edge_width_ratio):
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    instances_dir = os.path.join(scene_dir, "instances")
    bg_path = os.path.join(scene_dir, "bg.png")

    if not os.path.isdir(instances_dir) or not os.path.exists(bg_path):
        return "missing_input"

    inst_names = sorted(os.listdir(instances_dir))
    if not inst_names:
        return "no_instances"

    bg_resized = Image.open(bg_path).convert("RGB").resize((image_size, image_size))
    bg_arr = np.array(bg_resized).astype(np.float32)

    composite = bg_arr.copy()

    for inst_name in inst_names:
        inst_dir = os.path.join(instances_dir, inst_name)
        mask_arr, result_arr = harmonize_instance(
            pipe, bg_resized, inst_dir, image_size, num_inference_steps, edge_width_ratio
        )
        mask_3ch = mask_arr[..., None]
        composite = mask_3ch * result_arr + (1 - mask_3ch) * composite

    final_img = Image.fromarray(composite.astype(np.uint8))

    scene_out_dir = os.path.join(out_dir, scene_id)
    os.makedirs(scene_out_dir, exist_ok=True)
    final_img.save(os.path.join(scene_out_dir, "result.png"))

    scene_meta_path = os.path.join(scene_dir, "meta.json")
    if os.path.exists(scene_meta_path):
        with open(scene_meta_path) as f:
            meta = json.load(f)
        with open(os.path.join(scene_out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True,
                         help="build_synthetic_pairs_scene.py'nin uretttigi out_dir")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="BiliSakura/DiffusionSat-Single-512")
    parser.add_argument("--discriminator_path", type=str, default="./checkpoints/discriminator.pt")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--edge_width_ratio", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--limit", type=int, default=0, help="0 = hepsi, test icin kucuk sayi")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_scenes = sorted([
        d for d in os.listdir(args.scene_dir)
        if os.path.isdir(os.path.join(args.scene_dir, d))
    ])
    if args.limit > 0:
        all_scenes = all_scenes[:args.limit]
    print(f"Toplam sahne: {len(all_scenes)}")

    todo = []
    already_done = 0
    for scene_id in all_scenes:
        result_path = os.path.join(args.out_dir, scene_id, "result.png")
        if os.path.exists(result_path):
            already_done += 1
        else:
            todo.append(scene_id)
    print(f"Zaten islenmis (atlanacak): {already_done}")
    print(f"Islenecek: {len(todo)}")

    if not todo:
        print("Yapilacak bir sey yok.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    print(f"Model yukleniyor ({args.model_id}, device={device})...")
    discriminator = ResNet4ch.from_pretrained(args.discriminator_path, device=device)
    pipe = HarmoniDiffPipeline.from_pretrained_custom(
        args.model_id, discriminator=discriminator, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype]
    ).to(device)
    print("Model hazir.\n")

    stats = {"ok": 0, "missing_input": 0, "no_instances": 0, "error": 0}
    errors = []
    start_time = time.time()

    for i, scene_id in enumerate(todo, 1):
        scene_dir = os.path.join(args.scene_dir, scene_id)
        try:
            status = process_scene(pipe, scene_dir, args.out_dir, args.image_size,
                                    args.num_inference_steps, args.edge_width_ratio)
            stats[status] += 1
        except Exception as e:
            stats["error"] += 1
            errors.append((scene_id, str(e)))
            print(f"HATA - {scene_id}: {e}")

        if i % args.log_every == 0 or i == len(todo):
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = avg * (len(todo) - i)
            print(f"[{i}/{len(todo)}] ok={stats['ok']} error={stats['error']} "
                  f"| gecen={elapsed/60:.1f}dk | kalan_tahmini={remaining/60:.1f}dk")

    print("\n--- Ozet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print(f"\nHata alan sahneler (ilk 10):")
        for scene_id, msg in errors[:10]:
            print(f"  {scene_id}: {msg}")
    print(f"\nToplam tamamlanan: {already_done + stats['ok']} / {len(all_scenes)}")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
