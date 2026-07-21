"""
outputs/synthetic_pairs_k5/{scene}/instances/{inst}/meta.json dosyalarina,
HarmoniDiff'in pipe() cagrisinin ihtiyac duydugu alanlari (bg_prompt, fg_prompt,
longitude, latitude, bg_gsd, fg_gsd, cloud_cover, year, month, day) ekler.

Yeni klasor yapisini (scene/instances/inst) tarar - eski duz yapiyi
({scene}_inst{N}/meta.json) ARAMAZ, guncel pipeline (build_synthetic_pairs_scene.py)
ile uyumludur.

Kullanim:
    python enrich_meta.py --synthetic_dir outputs/synthetic_pairs_k5
"""
import argparse
import json
import os
import re

EVENT_INFO = {
    "turkey-earthquake":  {"country": "Turkey",  "longitude": 37.0,  "latitude": 37.2,  "gsd": 0.325, "year": 2023, "month": 2,  "day": 6},
    "morocco-earthquake": {"country": "Morocco", "longitude": -7.9,  "latitude": 31.1,  "gsd": 0.375, "year": 2023, "month": 9,  "day": 8},
    "noto-earthquake":    {"country": "Japan",   "longitude": 137.0, "latitude": 37.4,  "gsd": 0.5,   "year": 2024, "month": 1,  "day": 1},
    "haiti-earthquake":   {"country": "Haiti",   "longitude": -73.8, "latitude": 18.4,  "gsd": 0.48,  "year": 2021, "month": 8,  "day": 14},
}


def extract_event(scene_id):
    return re.sub(r'_\d+$', '', scene_id)


def build_prompts(country, class_name):
    bg_prompt = f"a SAR satellite image of buildings after an earthquake in {country}"
    fg_prompt = f"a SAR satellite image of a {class_name} building after an earthquake in {country}"
    return bg_prompt, fg_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", type=str, required=True)
    args = parser.parse_args()

    scene_dirs = sorted(os.listdir(args.synthetic_dir))
    n_ok, n_skip, n_unknown_event = 0, 0, 0
    unknown_events = set()

    for scene_id in scene_dirs:
        scene_path = os.path.join(args.synthetic_dir, scene_id)
        instances_path = os.path.join(scene_path, "instances")
        if not os.path.isdir(instances_path):
            n_skip += 1
            continue

        event_key = extract_event(scene_id)
        if event_key not in EVENT_INFO:
            unknown_events.add(event_key)
            n_unknown_event += 1
            continue

        info = EVENT_INFO[event_key]

        for inst_name in sorted(os.listdir(instances_path)):
            meta_path = os.path.join(instances_path, inst_name, "meta.json")
            if not os.path.isfile(meta_path):
                n_skip += 1
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            class_name = meta.get("class_name", "damaged")
            bg_prompt, fg_prompt = build_prompts(info["country"], class_name)

            meta.update({
                "country": info["country"],
                "longitude": info["longitude"],
                "latitude": info["latitude"],
                "bg_gsd": info["gsd"],
                "fg_gsd": info["gsd"],
                "cloud_cover": 0.0,
                "year": info["year"],
                "month": info["month"],
                "day": info["day"],
                "bg_prompt": bg_prompt,
                "fg_prompt": fg_prompt,
            })

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            n_ok += 1

    print(f"Guncellenen meta.json (instance): {n_ok}")
    print(f"Atlanan (meta.json yok/instances yok): {n_skip}")
    if n_unknown_event:
        print(f"Bilinmeyen event: {n_unknown_event} instance, event'ler: {unknown_events}")


if __name__ == "__main__":
    main()
