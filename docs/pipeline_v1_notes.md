# SynDataGen — Sentetik Hasar Verisi Üretim Pipeline'ı (v1)

## Amaç

BRIGHT veri setindeki post-event SAR görüntülerinden, sağlam (intact) binaların
üzerine gerçek damaged/destroyed bina crop'larını HarmoniDiff-RS ile harmonize
ederek yapıştırıp, class imbalance'ı (intact ~%83, damaged ~%11, destroyed ~%6.5)
azaltacak sentetik eğitim verisi üretmek.

Bu doküman v1'in (sahne başına 1 instance) tasarım kararlarını, veri akışını ve
bilinen kısıtları özetler.

---

## 1. Veri kaynakları

| Kaynak | İçerik | Konum |
|---|---|---|
| `raw/images/` | Post-event SAR arka plan görüntüleri (`.tif`), suffix `_post_disaster.tif` | `data/raw/images/` |
| `raw/target/` | Çok-sınıflı hasar mask'ı (`.tif`), suffix `_building_damage.tif` | `data/raw/target/` |
| `damagedbuildings/` | Önceden kesilmiş damaged/destroyed bina crop'ları (`.png`) + CSV metadata | `data/damagedbuildings/` |

**Mask encoding (BRIGHT resmi):** `0=background, 1=intact, 2=damaged, 3=destroyed`
(ESSD makalesindeki istatistiklerle doğrulandı: piksel bazında ~%82.8 intact,
~%10.7 damaged, ~%6.5 destroyed.)

**CSV şeması** (`damaged_buildings.csv`):
```
image_name, building_id, damage_class, class_name,
row_min, row_max, col_min, col_max, centroid_row, centroid_col,
pixel_area, saved_crop_path
```
- `row_min/row_max/col_min/col_max` = bbox (row=y, col=x)
- `pixel_area` = binanın gerçek piksel alanı (bbox alanı değil, düzensiz footprint)
- `saved_crop_path` = crop dosyasının kayıt anındaki yolu (klasör taşınmışsa
  script bunu görmezden gelip sadece dosya adını `--crops_dir` içinde arar)

**Kapsanan event'ler (bu turda kopyalanan):** turkey-earthquake, morocco-earthquake,
noto-earthquake, haiti-earthquake (BRIGHT'taki 14 event'in earthquake alt kümesi).

---

## 2. Tasarım kararları (v1)

Bu kararlar, geliştirme sürecinde soru-cevap yoluyla netleştirildi:

1. **Encoding:** `undamaged_value=1` (BRIGHT resmi standardı).
2. **Yapıştırma yeri:** Sağlam (intact=1) bina instance'ları `connected components`
   ile mask'tan otomatik çıkarılır — manuel seçim yok.
3. **Crop kaynağı — AYNI SAHNE KISITLAMASI:** Bir sahnedeki instance, **sadece o
   sahnenin kendi** CSV satırlarındaki (`image_name` eşleşen) crop'lardan biriyle
   eşleştirilir. Başka bir sahneden/event'ten crop **asla** kullanılmaz. Sebep:
   farklı event'ler farklı sensör kombinasyonlarına sahip (örn. Turkey=Maxar+
   Capella+Umbra, Noto=GSI+Umbra) — radyometrik/doku tutarlılığını korumak için.
4. **Boyut eşleştirme kriteri:** `pixel_area` (gerçek footprint alanı, bbox alanı
   değil) bazında en yakın komşu. Bir sahnede birden fazla intact instance varsa,
   **instance seçimi + crop eşleştirmesi birlikte optimize edilir** (o sahnenin
   crop'ları arasında en iyi boyut eşleşmesini veren instance otomatik seçilir).
5. **Sahne başına hacim:** v1'de **sadece 1 instance** dönüştürülür (bir sahnede
   birden fazla sağlam bina olsa bile). v2'de bu K>1'e genişletilecek (bkz. §6).
6. **Reuse:** v1'de zaten aynı-sahne kısıtlaması nedeniyle crop çakışması mümkün
   değil (her crop tek bir sahneye ait).
7. **Split/leakage:** Şu an train/val/test split ayrımı uygulanmıyor (kullanıcı
   kararı — şimdilik önemli değil, ileride event-bazlı cross-evaluation
   protokolüne uyulacaksa eklenmesi gerekecek).
8. **Ground truth güncelleme:** Seçilen instance'ın piksel değeri, orijinal mask
   kopyasında (`updated_label.tif`) artık `1` değil, yapıştırılan crop'un gerçek
   `damage_class`'ı (`2` veya `3`) olacak şekilde güncellenir — görüntü ile label
   tutarlılığı için zorunlu.

---

## 3. Pipeline akışı

```
data/raw/target/{scene}_building_damage.tif  ─┐
data/raw/images/{scene}_post_disaster.tif     ─┤─▶ build_synthetic_pairs.py ─▶ outputs/synthetic_pairs/{scene}_inst{NNNN}/
data/damagedbuildings/{crop}.png + CSV        ─┘         │                        ├── bg.png
                                                            │                        ├── mask_location.png
                                                            │                        ├── fg.png
                                                            │                        ├── updated_label.tif
                                                            │                        └── meta.json
                                                            ▼
                                                  (sonraki adım, henüz yazılmadı)
                                                  outputs/harmonized/{scene}_inst{NNNN}/
                                                        ├── result.png   ← HarmoniDiff çıktısı
                                                        └── ...
```

### Script: `scripts/build_synthetic_pairs.py`
```bash
python build_synthetic_pairs.py \
    --csv_path data/damagedbuildings/damaged_buildings.csv \
    --masks_dir data/raw/target \
    --images_dir data/raw/images \
    --crops_dir data/damagedbuildings \
    --mask_suffix _building_damage.tif \
    --bg_suffix _post_disaster.tif \
    --out_dir outputs/synthetic_pairs \
    --seed 42
```

Her sahne için:
1. Mask + bg dosyası var mı kontrol edilir (yoksa `missing_file`).
2. Mask'tan intact instance'lar çıkarılır (yoksa `no_intact_instances`).
3. O sahnenin CSV'deki kendi crop'ları arasında en iyi boyut eşleşmesi aranır
   (crop yoksa `pool_empty`).
4. Bulunursa: `bg.png / mask_location.png / fg.png / updated_label.tif / meta.json`
   üretilir (`ok`).
5. Beklenmeyen bir hata olursa sahne atlanır, batch durmaz (`error`).

---

## 4. v1 sonuçları (gerçekleşen çalıştırma)

```
Toplam CSV satırı: 31,236 (tüm event'ler)
İşlenen sahne sayısı: 1,497 (CSV'deki benzersiz image_name)
  ok:                  560
  missing_file:        937   (earthquake dışı event'ler, kopyalanmadı - beklenen)
  no_intact_instances:   0
  pool_empty:            0
  error:                 0
```

**Event bazlı kırılım (üretilen 560'ın kaynağı):**

| Event | Kopyalanan tile (BRIGHT resmi) | CSV'de damage içeren sahne | Oran |
|---|---|---|---|
| turkey-earthquake | 1,114 | 466 | %41.8 |
| noto-earthquake | 79 | 40 | %50.6 |
| haiti-earthquake | 73 | 29 | %39.7 |
| morocco-earthquake | 567 | 25 | **%4.4** ⚠️ |

**Bilinen anomali:** Morocco-earthquake oranı diğerlerinden belirgin şekilde
düşük. Olası nedenler: (a) hasar coğrafi olarak yoğunlaşmış (Atlas Dağları
kırsalı), tile'ların çoğu etkilenmemiş bölgede; (b) crop/CSV üretim sürecinde
Morocco için bir eksiklik. **Henüz doğrulanmadı, v2 öncesi araştırılabilir.**

---

## 5. Bilinen kısıtlar / henüz çözülmemiş konular

1. **HarmoniDiff prompt/metadata eksikliği:** `pipe()` çağrısı `bg_prompt`,
   `fg_prompt`, ve coğrafi/zamansal metadata (`longitude, latitude, bg_gsd,
   cloud_cover, year, month, day`) bekliyor. Bizim `meta.json`'umuzda bu alanlar
   yok — HarmoniDiff'in demo verisiyle test edilip nasıl doldurulacağı
   araştırılıyor (bkz. `notebooks/00_harmonidiff_demo_test.ipynb`).
2. **fg crop'larının alpha/şeffaflık durumu netleşmedi** — crop'ların gerçek
   bina silüetini mi (şeffaf arka plan), yoksa düz dikdörtgen bbox'ı mı
   içerdiği doğrulanmadı. Şu anki kod `.convert("RGB")` ile olası alpha
   kanalını siliyor; eğer crop'lar gerçekten silüet içeriyorsa bu bilgi
   kaybediliyor demektir.
3. **Sınır (border) instance kontrolü yapılmadı** — tile kenarına değen,
   kırpılmış bina parçalarının instance olarak yanlışlıkla seçilip
   seçilmediği doğrulanmadı.
4. **Split/leakage kısıtı yok** — train/val/test veya event-bazlı
   cross-evaluation ayrımı bu pipeline'a henüz dahil edilmedi.

---

## 6. v2 için planlanan genişletme

- **Sahne başına K>1 instance** (aynı-sahne kısıtlaması korunarak) — K, o
  sahnenin kendi crop sayısıyla doğal olarak sınırlı olacak. Genişletmeden
  önce `df.groupby("image_name").size()` dağılımına bakılıp gerçekçi bir K
  belirlenecek.
- HarmoniDiff çıktı kalitesi değerlendirildikten sonra, düşük kaliteli
  harmonizasyonları eleyecek bir filtre (discriminator skoru?) eklenmesi
  değerlendirilecek.

---

## 7. Dosya/script envanteri

| Dosya | Amaç | Durum |
|---|---|---|
| `scripts/inspect_mask.py` | Tek bir mask tif'in sınıf encoding'ini doğrular | Kullanılıyor |
| `scripts/build_synthetic_pairs.py` | v1 ana üretim script'i | Kullanılıyor (güncel) |
| `scripts/build_pasting_inputs.py` | Tek-sahne prototip (ilk versiyon) | Referans, aktif kullanılmıyor |
| `notebooks/BDI_pipeline_check.ipynb` | `build_synthetic_pairs.py`'yi çalıştırır + çıktıyı görsel kontrol eder | Kullanılıyor |
| `notebooks/00_harmonidiff_demo_test.ipynb` | HarmoniDiff'in kendi demo verisiyle tekli testi | Kullanılıyor (keşif aşaması) |
| `models/HarmoniDiff/` | HarmoniDiff-RS repo'su (pipeline_harmonidiff.py, melike.py, checkpoints/) | Bir sonraki adımda kullanılacak |
