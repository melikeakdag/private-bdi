import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import Resize

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


def load_instance(inst_dir):
    bg = Image.open(os.path.join(inst_dir, "bg.png")).convert("RGB")
    mask = Image.open(os.path.join(inst_dir, "mask_location.png")).convert("L")
    fg = Image.open(os.path.join(inst_dir, "fg.png")).convert("RGB")
    with open(os.path.join(inst_dir, "meta.json")) as f:
        meta = json.load(f)
    return bg, mask, fg, meta


def load_model_from_config(config, ckpt, device):
    print(f"Model yukleniyor: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    return model


def get_tensor():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_tensor_clip():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                          (0.26862954, 0.26130258, 0.27577711)),
    ])


@torch.inference_mode()
def rspaint_generate(model, sampler, bg_img, mask_img, ref_img, scale, ddim_steps, n_samples, device):
    img_p = bg_img.resize((512, 512))
    image_tensor = get_tensor()(img_p).unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device)

    ref_p = ref_img.resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_p).unsqueeze(0).repeat(n_samples, 1, 1, 1).to(device)

    mask = np.array(mask_img.resize((512, 512), Image.NEAREST))[None, None]
    mask = 1 - mask.astype(np.float32) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask).repeat(n_samples, 1, 1, 1).to(device)

    uc = model.learnable_vector.repeat(n_samples, 1, 1) if scale != 1.0 else None
    c = model.proj_out(model.get_learned_conditioning(ref_tensor))

    inpaint_image = image_tensor * mask_tensor
    z_inpaint = model.get_first_stage_encoding(model.encode_first_stage(inpaint_image)).detach()
    test_model_kwargs = {
        "inpaint_image": z_inpaint,
        "inpaint_mask": Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(mask_tensor),
    }

    samples_ddim, _ = sampler.sample(
        S=ddim_steps, conditioning=c, batch_size=n_samples, shape=[4, 64, 64],
        verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc,
        eta=0.0, x_T=None, test_model_kwargs=test_model_kwargs,
    )
    x = model.decode_first_stage(samples_ddim)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    return x.cpu().permute(0, 2, 3, 1).numpy()


def pick_best_sample(model, samples, ref_img, device):
    img_preprocessor = model.cond_stage_model.preprocess
    ref_feat = model.cond_stage_model.get_visual_clip_features(
        img_preprocessor(ref_img).unsqueeze(0).to(device)
    )
    best_idx, best_score = 0, -1.0
    for i in range(len(samples)):
        result = Image.fromarray((samples[i] * 255).astype(np.uint8))
        result_feat = model.cond_stage_model.get_visual_clip_features(
            img_preprocessor(result).unsqueeze(0).to(device)
        )
        score = F.cosine_similarity(ref_feat, result_feat, dim=-1).item()
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx, best_score


def harmonize_one_rspaint(model, sampler, inst_dir, out_dir, scale, ddim_steps, n_samples, device):
    bg, mask, fg, meta = load_instance(inst_dir)

    samples = rspaint_generate(model, sampler, bg, mask, fg, scale, ddim_steps, n_samples, device)
    best_idx, best_score = pick_best_sample(model, samples, fg, device)
    result_img = Image.fromarray((samples[best_idx] * 255).astype(np.uint8)).resize(bg.size)

    inst_name = os.path.basename(inst_dir.rstrip("/"))
    inst_out_dir = os.path.join(out_dir, inst_name)
    os.makedirs(inst_out_dir, exist_ok=True)
    result_img.save(os.path.join(inst_out_dir, "result_raw.png"))
    shutil.copy(os.path.join(inst_dir, "meta.json"), os.path.join(inst_out_dir, "meta.json"))
    with open(os.path.join(inst_out_dir, "clip_similarity.txt"), "w") as f:
        f.write(f"best_sample_idx={best_idx}\nclip_cosine_similarity={best_score:.5f}\n")

    print(f"OK: {inst_name} -> {inst_out_dir}/result_raw.png (clip_sim={best_score:.4f})")
    return inst_out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--instances", type=str, nargs="+", required=True)
    parser.add_argument("--config_path", type=str, default="configs/rs_remoteclip.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sd_inpaint_samrs_ep74.ckpt")
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250110)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = 0 if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt_path, device)
    sampler = PLMSSampler(model)
    print("Model hazir.")

    for inst_name in args.instances:
        inst_dir = os.path.join(args.synthetic_dir, inst_name)
        if not os.path.isdir(inst_dir):
            print(f"UYARI: klasor bulunamadi, atlaniyor: {inst_dir}")
            continue
        try:
            harmonize_one_rspaint(model, sampler, inst_dir, args.out_dir,
                                   args.scale, args.ddim_steps, args.n_samples, device)
        except Exception as e:
            print(f"HATA - {inst_name}: {e}")


if __name__ == "__main__":
    main()
