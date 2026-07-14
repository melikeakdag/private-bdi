"""
Mask tif dosyasindaki piksel deger dagilimini inceler ve BRIGHT'in resmi
kodlamasiyla (0=bg,1=intact,2=damaged,3=destroyed) karsilastirir.

Kullanim:
    python inspect_mask.py --mask_path /path/to/scene_mask.tif
"""
import argparse
import numpy as np
import rasterio
from scipy import ndimage

# BRIGHT resmi dagilim (ESSD makalesi, Fig 5d) - referans olarak
EXPECTED_RATIOS = {"intact": 0.828, "damaged": 0.107, "destroyed": 0.065}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", type=str, required=True)
    args = parser.parse_args()

    with rasterio.open(args.mask_path) as src:
        mask = src.read(1)
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    values, counts = np.unique(mask, return_counts=True)
    total_building_px = counts[values != 0].sum() if 0 in values else counts.sum()

    print("\nPiksel deger dagilimi:")
    for v, c in zip(values, counts):
        pct_all = 100 * c / mask.size
        pct_building = 100 * c / total_building_px if v != 0 and total_building_px > 0 else None
        line = f"  deger={int(v):>4}  piksel={c:>9}  toplamin %{pct_all:.2f}"
        if pct_building is not None:
            line += f"  (bina pikselinin %{pct_building:.2f})"
        print(line)

    print("\nBeklenen (BRIGHT resmi ortalama): intact ~%82.8, damaged ~%10.7, destroyed ~%6.5")
    print("-> Yukaridaki 'bina pikselinin %' sutunuyla en yakin esleseni bul,")
    print("   o deger o sinifa karsilik gelir. (Tek sahne icin sapma normaldir,")
    print("   ama genelde en buyuk pay 'intact', en kucuk pay 'destroyed' olur.)")

    print("\nHer deger icin bagli bilesen (instance) sayisi (0 haric):")
    for v in values:
        if v == 0:
            continue
        binary = (mask == v).astype(np.uint8)
        labeled, n = ndimage.label(binary)
        if n > 0:
            sizes = ndimage.sum(binary, labeled, range(1, n + 1))
            print(f"  deger={int(v)}: {n} instance, ort_boyut={sizes.mean():.0f}px, "
                  f"min={sizes.min():.0f}, max={sizes.max():.0f}")
        else:
            print(f"  deger={int(v)}: instance yok")


if __name__ == "__main__":
    main()
