import os
import argparse
import time
import random
import numpy as np
import rasterio
from PIL import Image
import torch
import json

from pipeline_harmonidiff import HarmoniDiffPipeline
from discrimination import ResNet4ch


def read_tif_as_pil(path):
    """SAR/optik tif dosyasını okuyup 8-bit RGB PIL görüntüsüne çevirir."""
    with rasterio.open(path) as src:
        n_bands = src.count
        if n_bands >= 3:
            arr = src.read([1, 2, 3]).astype(float)  # (3, H, W)
            arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
        else:
            band = src.read(1).astype(float)  # (H, W)
            p2, p98 = np.percentile(band, (2, 98))
            band_stretched = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
            arr = np.stack([band_stretched] * 3, axis=-1)  # (H, W, 3)
            arr_8bit = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr_8bit, mode='RGB')

    # çok bantlı (optik) görüntüler için ayrı stretch
    p2 = np.percentile(arr, 2, axis=(0, 1))
    p98 = np.percentile(arr, 98, axis=(0, 1))
    arr_stretched = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
    arr_8bit = (arr_stretched * 255).astype(np.uint8)
    return Image.fromarray(arr_8bit, mode='RGB')


def main():

    parser = argparse.ArgumentParser(description="inference script for harmonydiff pipeline")

    parser.add_argument('--model_id', type=str, default="BiliSakura/DiffusionSat-Single-512", )
    parser.add_argument('--discriminator_path', type=str, default='./checkpoints/discriminator.pt')
    parser.add_argument('--data_path', type=str, default='./demo')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--edge_width_ratio', type=float, default=0.1)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'],)
    parser.add_argument('--visualization_path', type=str, default='./visualization')
    parser.add_argument('--seed', type=int, default=None,
                         help='rastgele konumu sabitlemek için (tekrar üretilebilirlik)')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.visualization_path is not None:
        os.makedirs(args.visualization_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16
    }

    discriminator = ResNet4ch.from_pretrained(args.discriminator_path, device=device)
    pipe = HarmoniDiffPipeline.from_pretrained_custom(args.model_id, discriminator=discriminator, trust_remote_code=True,
                                                      torch_dtype=dtype_map[args.dtype]).to(device)


    with open(f'{args.data_path}/meta.json', 'r') as fn:
        meta = json.load(fn)

    background = meta['background']
    foreground = meta['foreground']

    metadata = [meta['longitude'], meta['latitude'], meta['bg_gsd'], meta['cloud_cover'], meta['year'], meta['month'], meta['day']]

    # ---- TIF OKUMA ----
    bg_img = read_tif_as_pil(f'{args.data_path}/{background}.tif').resize((args.image_size, args.image_size))
    fg_img = read_tif_as_pil(f'{args.data_path}/{foreground}.png')

    # ---- fg kendi boyutunu korur, bbox kullanılmıyor ----
    w, h = fg_img.size

    if w > args.image_size or h > args.image_size:
        scale = min(args.image_size / w, args.image_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        fg_img = fg_img.resize((new_w, new_h))
        w, h = new_w, new_h
        print(f"fg_img bg'den büyüktü, {new_w}x{new_h} boyutuna küçültüldü.")

    # ---- rastgele konum ----
    max_x = args.image_size - w
    max_y = args.image_size - h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    print(f"fg boyutu: {w}x{h}, yapıştırma konumu: ({x},{y})")

    init = bg_img.copy()
    init.paste(fg_img, (x, y))

    if args.visualization_path is not None:
        bg_img.save(f'{args.visualization_path}/bg.png')
        init.save(f'{args.visualization_path}/init.png')

    px, py, pw, ph = [cord / args.image_size for cord in [x, y, w, h]]

    start = time.time()
    image = pipe(bg_prompt=meta['bg_prompt'], fg_prompt=meta['fg_prompt'], bg_image=bg_img, fg_image=fg_img, xy=[px, py],
                    metadata=metadata, num_inference_steps=20, height=args.image_size, width=args.image_size,
                        edge_width_ratio=args.edge_width_ratio, viz_checkpoints=args.visualization_path)
    end = time.time()
    print(f'Inference completed in {end-start:.2f} seconds.')
    image.save(f'{args.visualization_path}/result.png')

    used_bbox = {"x": x, "y": y, "w": w, "h": h}
    with open(f'{args.visualization_path}/used_bbox.json', 'w') as f:
        json.dump(used_bbox, f, indent=2)


if __name__ == "__main__":
    main()