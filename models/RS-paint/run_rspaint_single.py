"""
run_rspaint_single.py
======================
run_harmonidiff_single.py'nin RS-Paint'e UYARLANMIS hali. Ayni instance
klasor yapisini (bg.png / mask_location.png / fg.png / meta.json) okur,
sadece MODEL COGRISI HarmoniDiff yerine RS-Paint (Paint-by-Example tabanli
Stable Diffusion inpainting + RemoteCLIP) olur.
"""
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
    bg, mask, fg,
