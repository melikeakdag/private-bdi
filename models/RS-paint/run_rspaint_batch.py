"""
run_rspaint_batch.py
=====================
run_rspaint_single.py'nin TOPLU (batch) hali. --instances ile tek tek isim
vermek yerine, --synthetic_dir altindaki TUM instance klasorlerini bulup
sirayla isler. Model SADECE BIR KEZ yuklenir (tekil script'te de boyleydi,
ama tekil script'i N kere cagirirsan model N kere yuklenir - bu, GPU'ya
gereksiz tekrar yukleme yapmadan hepsini isler).

OZELLIKLER:
  - Zaten islenmis (out_dir'da result.png'si olan) instance'lari ATLAR
    -> Colab kopup tekrar baslatirsan kaldigin yerden devam eder (resume).
  - Eksik dosyali (bg.png/mask_location.png/fg.png/meta.json yok) klasorleri
    atlar, hata vermez.
  - Her instance ayri try/except -> biri patlarsa butun batch durmaz,
    hatalar sonunda ozetlenir.
  - --limit ile once kucuk bir alt kumede test etmeni saglar.
  - --shuffle ile rastgele siraya sokar (COK BUYUK setlerde ilk N'i degil
    cesitli sahnelerden ornek gormek icin faydali).

Kullanim (rs-paint repo kokunden):
    python run_rspaint_batch.py \
        --synthetic_dir /content/drive/MyDrive/SynDataGen/outputs/synthetic_pairs \
        --out_dir /content/drive/MyDrive/SynDataGen/outputs/rspaint_results \
        --limit 20              # once 20 tanesiyle test et
        # --limit vermezsen TUMUNU isler
"""
import argparse
import json
import os
import random
import shutil
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
    result_raw_img = Image.fromarray((samples[best_idx] * 255).astype(np.uint8)).resize(bg.size)

    # EKLENDI: run_harmonidiff_single.py ile AYNI post-process - mask disini
    # orijinal bg'ye geri donduruyoruz (VAE reconstruction gurultusunu change
    # detection etiketiyle tutarli hale getirmek icin, bkz. run_rspaint_single.py)
    mask_resized = mask.resize(bg.size, resample=Image.NEAREST)
    mask_arr = (np.array(mask_resized) > 127).astype(np.float32)[..., None]
    result_arr = np.array(result_raw_img.convert("RGB")).astype(np.float32)
    bg_arr = np.array(bg.convert("RGB")).astype(np.float32)
    final_arr = (mask_arr * result_arr + (1 - mask_arr) * bg_arr).astype(np.uint8)
    final_img = Image.fromarray(final_arr)

    inst_name = os.path.basename(inst_dir.rstrip("/"))
    inst_out_dir = os.path.join(out_dir, inst_name)
    os.makedirs(inst_out_dir, exist_ok=True)
    result_raw_img.save(os.path.join(inst_out_dir, "result_raw.png"))   # ham (VAE gurultulu) cikti
    final_img.save(os.path.join(inst_out_dir, "result.png"))            # NIHAI (mask-disi korunmus) cikti
    shutil.copy(os.path.join(inst_dir, "meta.json"), os.path.join(inst_out_dir, "meta.json"))
    with open(os.path.join(inst_out_dir, "clip_similarity.txt"), "w") as f:
        f.write(f"best_sample_idx={best_idx}\nclip_cosine_similarity={best_score:.5f}\n")

    return inst_out_dir, best_score


def discover_instances(synthetic_dir):
    """synthetic_dir altinda gecerli (bg/mask/fg/meta iceren) instance klasorlerini bulur."""
    valid, skipped_incomplete = [], []
    for name in sorted(os.listdir(synthetic_dir)):
        inst_dir = os.path.join(synthetic_dir, name)
        if not os.path.isdir(inst_dir):
            continue
        required = ["bg.png", "mask_location.png", "fg.png", "meta.json"]
        if all(os.path.exists(os.path.join(inst_dir, r)) for r in required):
            valid.append(name)
        else:
            skipped_incomplete.append(name)
    return valid, skipped_incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="configs/rs_remoteclip.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sd_inpaint_samrs_ep74.ckpt")
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250110)
    parser.add_argument("--limit", type=int, default=None,
                         help="Sadece ilk N instance'i isle (test icin). Vermezsen TUMU islenir.")
    parser.add_argument("--shuffle", action="store_true",
                         help="Instance sirasini rastgele karistir (once --limit ile kucuk, cesitli bir alt kume gormek icin)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                         help="out_dir'da zaten result.png'si olan instance'lari atla (varsayilan: acik)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                         help="Zaten islenmis olsa da HEPSINI YENIDEN isle")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = 0 if torch.cuda.is_available() else "cpu"

    all_instances, skipped_incomplete = discover_instances(args.synthetic_dir)
    print(f"Bulunan gecerli instance: {len(all_instances)}")
    if skipped_incomplete:
        print(f"Eksik dosyali (bg/mask/fg/meta hepsi yok) atlanan klasor: {len(skipped_incomplete)}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_instances)

    todo = []
    already_done = 0
    for name in all_instances:
        result_path = os.path.join(args.out_dir, name, "result.png")
        if args.skip_existing and os.path.exists(result_path):
            already_done += 1
            continue
        todo.append(name)

    if already_done:
        print(f"Daha once islenmis (atlanan): {already_done}")

    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Bu calistirmada islenecek instance sayisi: {len(todo)}")
    if not todo:
        print("Islenecek instance yok, cikiliyor.")
        return

    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt_path, device)
    sampler = PLMSSampler(model)
    print("Model hazir.\n")

    ok_count, err_count = 0, 0
    errors = []
    t_start = time.time()

    for i, inst_name in enumerate(todo, 1):
        inst_dir = os.path.join(args.synthetic_dir, inst_name)
        t0 = time.time()
        try:
            inst_out_dir, score = harmonize_one_rspaint(
                model, sampler, inst_dir, args.out_dir,
                args.scale, args.ddim_steps, args.n_samples, device,
            )
            dt = time.time() - t0
            ok_count += 1
            print(f"[{i}/{len(todo)}] OK  {inst_name}  clip_sim={score:.4f}  ({dt:.1f}s)")
        except Exception as e:
            err_count += 1
            errors.append((inst_name, str(e)))
            print(f"[{i}/{len(todo)}] HATA {inst_name}: {e}")

        # her 10 instance'ta bir toplam ilerleme/sure tahmini goster
        if i % 10 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            avg = elapsed / i
            remaining = avg * (len(todo) - i)
            print(f"    -- ilerleme: {i}/{len(todo)}  ortalama={avg:.1f}s/instance  "
                  f"tahmini kalan sure={remaining/60:.1f} dk")

    print("\n--- Ozet ---")
    print(f"Basarili: {ok_count}")
    print(f"Hatali:   {err_count}")
    if errors:
        print("Hatali instance'lar (ilk 10):")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")
    print(f"Toplam sure: {(time.time() - t_start)/60:.1f} dk")
    print(f"Cikti klasoru: {args.out_dir}")


if __name__ == "__main__":
    main()
