import os
import argparse
import time
from PIL import Image
import torch
import json

from pipeline_harmonidiff import HarmoniDiffPipeline
from discrimination import ResNet4ch


def main():

    parser = argparse.ArgumentParser(description="inference script for harmonydiff pipeline")

    parser.add_argument('--model_id', type=str, default="BiliSakura/DiffusionSat-Single-512", )
    parser.add_argument('--discriminator_path', type=str, default='./checkpoints/discriminator.pt')
    parser.add_argument('--data_path', type=str, default='./demo')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--edge_width_ratio', type=float, default=0.1)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'],)
    parser.add_argument('--visualization_path', type=str, default='./visualization')


    args = parser.parse_args()

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

    bg_img = Image.open(f'{args.data_path}/{background}.png').convert('RGB').resize((args.image_size,args.image_size))
    fg_img = Image.open(f'{args.data_path}/{foreground}.png').convert('RGB')


    x,y,w,h = meta['bbox'] # x,y,w,h
    fg_img = fg_img.resize((w,h))

    init = bg_img.copy()
    init.paste(fg_img, (x,y))

    if args.visualization_path is not None:
        bg_img.save(f'{args.visualization_path}/bg.png')
        init.save(f'{args.visualization_path}/init.png')

    px,py,pw,ph = [cord/args.image_size for cord in [x,y,w,h]]

    start = time.time()
    image = pipe(bg_prompt=meta['bg_prompt'], fg_prompt=meta['fg_prompt'], bg_image=bg_img, fg_image=fg_img, xy=[px,py],
                    metadata=metadata, num_inference_steps=20, height=args.image_size, width=args.image_size,
                        edge_width_ratio=args.edge_width_ratio, viz_checkpoints=args.visualization_path)
    end = time.time()
    print(f'Inference completed in {end-start:.2f} seconds.')
    image.save(f'{args.visualization_path}/result.png')

if __name__ == "__main__":
    main()