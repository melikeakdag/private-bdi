# SynDataGen — Sentetik Hasar Verisi Üretim Pipeline'ı (v2)

## v1'den farkı

v1 dokümanında (bkz. `docs/pipeline_v1_notes.md`) sadece HarmoniDiff-RS ile
üretim ve tek-yöntemli çıktı vardı. v2'de:

1. **Aynı-sahne kısıtlaması düzeltildi** — v1'de crop eşleştirmesi global bir
   havuzdan yapılıyordu (bir sahnenin binası başka bir event'ten crop alabiliyordu).
   Bu, tasarım kararına aykırıydı ve düzeltildi: artık her sahne **sadece kendi**
   CSV satırlarındaki crop'lardan seçim yapıyor.
2. **İkinci bir harmonizasyon yöntemi eklendi: RSPaint** — HarmoniDiff-RS ile
   yan yana karşılaştırmalı değerlendirme yapıldı.
3. **Kapsamlı nicel değerlendirme pipeline'ı** (`inpainting_eval.py`) yazıldı —
   SSIM, PSNR, LPIPS, sınır (boundary) süreklilik analizi, yüksek-frekans
   (doku) korunumu, renk/harmonizasyon tutarlılığı, mask-dışı sızıntı kontrolü.
4. **HarmoniDiff'in `pipe()` çağrısına eksik olan metadata alanları** (bg_prompt,
   fg_prompt, longitude, latitude, bg_gsd, cloud_cover, year, month, day)
   `enrich_meta.py` ile eklendi (event bazlı gerçek tarih/GSD değerleri,
   BRIGHT makalesinden).
5. **Post-processing maskeleme netleşti:** HarmoniDiff'in `pipe()` çağrısı sadece
   dikdörtgen bbox'ı harmonize ediyor (bina siluetini bilmiyor) — bu yüzden
   `result_raw.png` (ham pipe çıktısı) ile bizim gerçek bina siluetimize
   (`mask_location.png`) göre maskelenmiş `result.png` ayrı tutuldu.

---

## 1. Düzeltme sonrası veri hacmi

Aynı-sahne kısıtlaması düzeltmesi sonrası üretim tekrarlandı:

```
v1 (hatalı, global pool):  560 instance
v2 (düzeltilmiş):          529 instance  (analiz edilebilen)
```

**Event dağılımı (v2):**

| Event | Instance sayısı |
|---|---|
| turkey-earthquake | 441 |
| noto-earthquake | 35 |
| haiti-earthquake | 28 |
| morocco-earthquake | 25 |

**Damage class dağılımı:** destroyed (296) > damaged (233)

**Mask alanı istatistikleri (512x512 canvas üzerinde, piksel):**

| İstatistik | Değer |
|---|---|
| min | 49 |
| medyan | 454 |
| ortalama | 629 |
| max | 5384 |

En büyük mask: `turkey-earthquake_00000602_inst0002` (5384 px, destroyed)
En küçük mask: `haiti-earthquake_00000049_inst0002` (49 px, destroyed)

---

## 2. Değerlendirme metodolojisi (`inpainting_eval.py`)

Her instance için, `bg.png` (orijinal) ile her yöntemin `result.png`'si
(HarmoniDiff ve RSPaint) karşılaştırılıyor. Hesaplanan metrikler:

| Metrik | Ne ölçüyor | Yön |
|---|---|---|
| `outside_mean` / `outside_leak_ratio` | Mask **dışındaki** piksellerin değişip değişmediği (post-process doğruluğu) | idealde 0 |
| `diff_mean` / `inside_mean` | Mask **içindeki** ortalama piksel farkı (bg vs result) | düşük=iyi (ama 0'a çok yakınsa "hiçbir şey değişmemiş" şüphesi de doğurur) |
| `boundary_ratio` | Mask sınırındaki gradyan sertliği, bg'ye kıyasla (dikiş izi göstergesi) | düşük=iyi |
| `hf_energy_ratio` | Mask içindeki yüksek-frekans (doku detayı) enerjisi, bg'ye kıyasla oran | ~1'e yakın=iyi (çok düşükse aşırı yumuşama) |
| `color_consistency` | Mask içi ile hemen dışındaki ortalama renk farkı | düşük=iyi |
| `ssim` / `psnr` | Standart görüntü benzerlik metrikleri (tüm görüntü üzerinden) | yüksek=iyi |
| `lpips` | Algısal (perceptual) benzerlik farkı | düşük=iyi |

**⚠️ Metodolojik uyarı:** `ssim`/`psnr`/`lpips` **tüm görüntü** üzerinden
hesaplanıyor, sadece mask bölgesi üzerinden değil. Bu metriklerle
`mask_area_px_512` arasında çok güçlü korelasyon var (`ssim`: r=-0.991,
`lpips`: r=0.833) — yani büyük bir bina değiştirildiğinde bu metrikler
**mekanik olarak** kötüleşiyor, düzenleme kalitesinden bağımsız olarak. Bu
metrikleri farklı instance'lar arasında mutlak kalite göstergesi olarak
yorumlarken dikkatli olunmalı; asıl adil karşılaştırma **aynı instance
üzerinde iki yöntemi kıyaslamak** (bu raporun yaptığı budur).

---

## 3. Sonuçlar — sızıntı kontrolü

Hem HarmoniDiff hem RSPaint için `outside_mean = 0.0` (tüm 529 instance'da) —
post-process maskeleme **her iki yöntemde de kusursuz çalışıyor**, mask
dışına hiçbir piksel sızmıyor.

---

## 4. Sonuçlar — HarmoniDiff vs RSPaint

**HarmoniDiff kazanma oranı** (529 instance üzerinden, %50 = berabere):

| Metrik | HarmoniDiff kazanma oranı | HarmoniDiff ort. | RSPaint ort. |
|---|---|---|---|
| `boundary_ratio` | **%88.8** | 0.843 | 0.944 |
| `psnr` | %56.9 | 39.81 | 39.57 |
| `color_consistency` | %55.6 | 33.74 | 38.08 |
| `diff_mean` | %54.8 | 0.416 | 0.415 |
| `inside_mean` | %54.8 | 162.40 | 163.81 |
| `ssim` | %53.9 | 0.9979 | 0.9979 |
| `lpips` | %37.8 | 0.0018 | 0.0015 |
| `hf_energy_ratio` | **%17.2** | 0.827 | 0.952 |

### Yorum: net bir kazanan yok, bir trade-off var

- **HarmoniDiff** kenar/sınır geçişinde (`boundary_ratio`) belirgin şekilde
  daha güçlü — dikiş izi bırakmadan yapıştırıyor (%88.8 örnekte RSPaint'ten
  daha iyi).
- **HarmoniDiff, doku/detay kaybı yaşıyor** (`hf_energy_ratio` 0.827, RSPaint
  0.952'ye kıyasla belirgin düşük) — aşırı yumuşatma eğiliminde. Bu, `lpips`'te
  (algısal fark) hafif dezavantaja da yansıyor (%37.8 kazanma oranı, yani
  RSPaint çoğu zaman algısal olarak biraz daha sadık).
- `ssim`, `psnr`, `diff_mean` gibi genel metriklerde neredeyse berabere.

**Pratik çıkarım:** İki yöntem birbirini tamamlayan güçlü/zayıf yönlere sahip.
Kenar sürekliliği öncelikliyse HarmoniDiff, doku/detay korunumu öncelikliyse
RSPaint tercih edilebilir. Hibrit bir yaklaşım (örn. HarmoniDiff'in kenar
geçişini + RSPaint'in iç doku üretimini birleştiren bir post-process) v3 için
değerlendirilebilir.

---

## 5. Manuel inceleme için işaretlenen örnekler

| Kategori | Instance | Değer | Not |
|---|---|---|---|
| En büyük mask | `turkey-earthquake_00000602_inst0002` | 5384 px | SSIM/LPIPS confound'unun en belirgin olduğu örnek |
| En küçük mask | `haiti-earthquake_00000049_inst0002` | 49 px | Çok küçük binalarda harmonizasyon davranışı |
| En yüksek `boundary_ratio` (HarmoniDiff) | `morocco-earthquake_00000493_inst0001` | 1.78 | Normalin çok üzerinde — dikiş izi şüphesi, gözle doğrulanmalı |
| En düşük `hf_energy_ratio` (HarmoniDiff) | `turkey-earthquake_00000283_inst0001` | 0.49 | Aşırı yumuşama şüphesi |

---

## 6. Bilinen kısıtlar (v2'de hâlâ açık)

1. **Sahne başına hâlâ sadece 1 instance** — v2'de K>1 genişletmesi henüz
   yapılmadı (v1 §6'da planlanmıştı, öncelik RSPaint entegrasyonu + kalite
   değerlendirmesi oldu).
2. **Morocco-earthquake anomalisi** (v1'de not edilen %4.4 düşük kapsama oranı)
   henüz araştırılmadı.
3. **fg crop'larının alpha/şeffaflık durumu** hâlâ netleşmedi (v1 §5.2).

---

## 7. v3 için olası yönler

- Sahne başına K>1 instance genişletmesi (v1'de planlanan)
- Üretilen sentetik verinin gerçek bir downstream model (bina hasar
  sınıflandırma) üzerinde eğitim/değerlendirme etkisinin ölçülmesi
