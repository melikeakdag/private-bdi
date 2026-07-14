"""
outputs/synthetic_pairs/{scene}_inst{N}/ icindeki bg.png/mask_location.png/fg.png/
meta.json'i okuyup HarmoniDiff ile harmonize eder. Ciktida iki dosya olur:
  - result_raw.png : pipe()'in dondugu HAM sonuc (dikdortgen bbox harmonize edilmis)
  - result.png     : bizim GERCEK bina siluetimize (mask_location.png) gore
                      post-process edilmis nihai sonuc (bbox disindaki her sey
                      orijinal bg'ye geri dondurulmus)

Bu script BATCH DEGIL - sadece 1-2 ornek uzerinde hizli dogrulama icin.
models/HarmoniDiff/ klasorunun ICINDE durmali (pipeline_harmonidiff.py ve
discrimination.py'yi relative import ediyor).

ONEMLI: meta.json'da bg_prompt/fg_prompt/longitude/latitude/bg_gsd/cloud_cover/
year/month/day alanlari olmali - yoksa once enrich_meta.py'yi calistir.

Kullanim:
    cd models/HarmoniDiff
    python run_harmonidiff_single.py \
        --synthetic_dir /content/drive/MyDrive/SAR/SynDataGen/outputs/synthetic_pairs \
        --out_dir /content/drive/MyDrive/SAR/SynDataGen/outputs/harmonized \
        --instances turkey-earthquake_00001030_inst0001 turkey-earthquake_00000933_inst0001
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from pipeline_harmonidiff import HarmoniDiffPipeline
from discrimination import ResNet4ch


def load_instance(inst_dir):
    bg = Image.open(os.path.join(inst_dir, "bg.png")).convert("RGB")
    mask = Image.open(os.path.join(inst_dir, "mask_location.png")).convert("L")
    fg = Image.open(os.path.join(inst_dir, "fg.png")).convert("RGB")
    with open(os.path.join(inst_dir, "meta.json")) as f:
        meta = json.load(f)
    return bg, mask, fg, meta


def harmonize_one(pipe, inst_dir, out_dir, image_size, num_inference_steps, edge_width_ratio):
    bg, mask, fg, meta = load_instance(inst_dir)

    required = ["bg_prompt", "fg_prompt", "longitude", "latitude", "bg_gsd",
                "cloud_cover", "year", "month", "day"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"meta.json eksik alan(lar): {missing} - once enrich_meta.py calistir")

    metadata = [meta["longitude"], meta["latitude"], meta["bg_gsd"], meta["cloud_cover"],
                meta["year"], meta["month"], meta["day"]]

    px, py = meta["xy_normalized"]

    bg_resized = bg.resize((image_size, image_size))

    # fg boyutunu normalize orandan hedef canvas (image_size) icin yeniden hesapla
    # (kaydedilen fg.png orijinal (buyuk) mask cozunurlugune gore boyutlanmisti)
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
        viz_checkpoints=None,  # ara adimlari KAYDETME - sadece nihai sonuc
    )

    # --- post-process: pipe SADECE dikdortgen bbox'i harmonize ediyor.
    # Bizim GERCEK bina siluetimizin (mask_location.png) DISINDA kalan her seyi
    # orijinal bg'ye geri donduruyoruz, boylece bbox'in bos koseleri bozulmuyor.
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

    print(f"OK: {inst_name} -> {inst_out_dir}/result.png")
    return inst_out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--instances", type=str, nargs="+", required=True,
                         help="Islenecek instance klasor adlari (orn. turkey-earthquake_00001030_inst0001)")
    parser.add_argument("--model_id", type=str, default="BiliSakura/DiffusionSat-Single-512")
    parser.add_argument("--discriminator_path", type=str, default="./checkpoints/discriminator.pt")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--edge_width_ratio", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    print(f"Model yukleniyor ({args.model_id}, device={device})...")
    discriminator = ResNet4ch.from_pretrained(args.discriminator_path, device=device)
    pipe = HarmoniDiffPipeline.from_pretrained_custom(
        args.model_id, discriminator=discriminator, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype]
    ).to(device)
    print("Model hazir.")

    for inst_name in args.instances:
        inst_dir = os.path.join(args.synthetic_dir, inst_name)
        if not os.path.isdir(inst_dir):
            print(f"UYARI: klasor bulunamadi, atlaniyor: {inst_dir}")
            continue
        try:
            harmonize_one(pipe, inst_dir, args.out_dir, args.image_size,
                          args.num_inference_steps, args.edge_width_ratio)
        except Exception as e:
            print(f"HATA - {inst_name}: {e}")


if __name__ == "__main__":
    main()
