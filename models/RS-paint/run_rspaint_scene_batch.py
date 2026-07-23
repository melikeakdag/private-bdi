"""
run_rspaint_scene_batch.py
===========================
run_harmonidiff_scene_batch.py'nin RS-Paint'e uyarlanmis hali. outputs/
synthetic_pairs_scene_test/{scene_id}/ altindaki HER instance'i BAGIMSIZ
olarak (orijinal bg uzerinde) RS-Paint ile inpaint eder, sonra hepsini TEK
bir result.png'de birlestirir (mask'lar ortusmedigi icin sorunsuz birlesir).

HarmoniDiff versiyonundan farklar:
  - enrich_meta.py alanlarina (bg_prompt/fg_prompt/lon/lat/gsd/tarih) IHTIYAC YOK.
  - fg, 224x224 CLIP-normalize ediliyor (bbox boyutuna resize YOK).
  - mask, RS-Paint'in kendi ldm kodu tarafindan GERCEKTEN kullaniliyor (bizim
    bina siluetimiz - HarmoniDiff'teki gibi sadece bbox degil).
  - n_samples uretilip RemoteCLIP cosine-similarity'e gore en iyisi secilir.

Cikti: outputs/rspaint_scene/{scene_id}/result.png (TEK foto, K hasarin hepsi)
       + outputs/rspaint_scene/{scene_id}/instance_scores.json (her instance'in
         clip_similarity skoru, hangi orneginin secildigi)

BU SCRIPT rs-paint REPO KOKUNDEN calistirilmali (ldm.*, segment_anything
relative/kurulu paketler uzerinden import ediliyor):
    cd rs-paint
    python run_rspaint_scene_batch.py \
        --scene_dir /path/to/outputs/synthetic_pairs_scene_test \
        --out_dir /path/to/outputs/rspaint_scene \
        --limit 5
"""
import argparse
import json
import os
import time

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


def harmonize_instance_rspaint(model, sampler, bg_resized, inst_dir, scale, ddim_steps, n_samples, device):
    """Tek bir instance'i, VERILEN (orijinal) bg uzerinde inpaint eder.
    O instance'in kendi maskesini, harmonize edilmis goruntuyu ve clip skorunu dondurur."""
    mask = Image.open(os.path.join(inst_dir, "mask_location.png")).convert("L")
    fg = Image.open(os.path.join(inst_dir, "fg.png")).convert("RGB")

    samples = rspaint_generate(model, sampler, bg_resized, mask, fg, scale, ddim_steps, n_samples, device)
    best_idx, best_score = pick_best_sample(model, samples, fg, device)
    result_img = Image.fromarray((samples[best_idx] * 255).astype(np.uint8)).resize(bg_resized.size)

    mask_resized = mask.resize(bg_resized.size, resample=Image.NEAREST)
    mask_arr = (np.array(mask_resized) > 127).astype(np.float32)
    result_arr = np.array(result_img.convert("RGB")).astype(np.float32)

    return mask_arr, result_arr, best_idx, best_score


def process_scene(model, sampler, scene_dir, out_dir, scale, ddim_steps, n_samples, device):
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    instances_dir = os.path.join(scene_dir, "instances")
    bg_path = os.path.join(scene_dir, "bg.png")

    if not os.path.isdir(instances_dir) or not os.path.exists(bg_path):
        return "missing_input"

    inst_names = sorted(os.listdir(instances_dir))
    if not inst_names:
        return "no_instances"

    bg_resized = Image.open(bg_path).convert("RGB").resize((512, 512))
    bg_arr = np.array(bg_resized).astype(np.float32)

    composite = bg_arr.copy()
    instance_scores = {}

    for inst_name in inst_names:
        inst_dir = os.path.join(instances_dir, inst_name)
        mask_arr, result_arr, best_idx, best_score = harmonize_instance_rspaint(
            model, sampler, bg_resized, inst_dir, scale, ddim_steps, n_samples, device
        )
        mask_3ch = mask_arr[..., None]
        composite = mask_3ch * result_arr + (1 - mask_3ch) * composite
        instance_scores[inst_name] = {"best_sample_idx": best_idx, "clip_similarity": round(best_score, 5)}

    final_img = Image.fromarray(composite.astype(np.uint8))

    scene_out_dir = os.path.join(out_dir, scene_id)
    os.makedirs(scene_out_dir, exist_ok=True)
    final_img.save(os.path.join(scene_out_dir, "result.png"))

    with open(os.path.join(scene_out_dir, "instance_scores.json"), "w") as f:
        json.dump(instance_scores, f, indent=2)

    scene_meta_path = os.path.join(scene_dir, "meta.json")
    if os.path.exists(scene_meta_path):
        with open(scene_meta_path) as f:
            meta = json.load(f)
        with open(os.path.join(scene_out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="configs/rs_remoteclip.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sd_inpaint_samrs_ep74.ckpt")
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250110)
    parser.add_argument("--limit", type=int, default=0, help="0 = hepsi, test icin kucuk sayi")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = 0 if torch.cuda.is_available() else "cpu"

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

    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt_path, device)
    sampler = PLMSSampler(model)
    print("Model hazir.\n")

    stats = {"ok": 0, "missing_input": 0, "no_instances": 0, "error": 0}
    errors = []
    start_time = time.time()

    for i, scene_id in enumerate(todo, 1):
        scene_dir = os.path.join(args.scene_dir, scene_id)
        try:
            status = process_scene(model, sampler, scene_dir, args.out_dir,
                                    args.scale, args.ddim_steps, args.n_samples, device)
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
