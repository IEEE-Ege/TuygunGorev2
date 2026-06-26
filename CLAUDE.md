# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje Özeti

TEKNOFEST 2026 Havacılıkta Yapay Zeka yarışması — **TUYGUN** takımı.

Yarışma iki ana görevden oluşur:
- **Görev 1:** Nesne tespiti (UAV görüntüsünden Taşıt/İnsan/UAP/UAI sınıflandırma + inilebilirlik)
- **Görev 2:** Pozisyon kestirimi (GPS sağlıksızken Visual Odometry ile 3D konum tahmini)

**Sadece `competition_repo/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` dosyası değiştirilmelidir.** Diğer framework dosyaları (connection_handler, constants, vb.) değiştirilmez.

## Proje Yapısı

```
Teknofest/
├── competition_repo/              # git clone ile gelen yarışma framework'ü
│   ├── TAKIM_BAGLANTI_ARAYUZU/
│   │   ├── main.py               # Entry point — doğrudan çalıştırılır
│   │   ├── requirements.txt      # python-decouple, requests, pillow, tqdm
│   │   ├── config/example.env    # .env şablonu (TEAM_NAME, PASSWORD, EVALUATION_SERVER_URL)
│   │   └── src/
│   │       ├── object_detection_model.py  # SADECE BU DOSYA DÜZENLENIR
│   │       ├── frame_predictions.py       # FramePredictions — tahmin container
│   │       ├── detected_object.py         # DetectedObject — bbox çıktı sınıfı
│   │       ├── detected_translation.py    # DetectedTranslation — pozisyon çıktı sınıfı
│   │       ├── constants.py               # classes / landing_statuses dict
│   │       └── connection_handler.py      # Sunucu iletişimi
│   └── Kamera_Kalibrasyon/
│       └── Kamera_Kalibrasyon_Parametreleri_2025.txt
├── 2026_HAVACILIKTA_YAPAY_ZEKA_TEKNIK_SARTNAME_TR_*.pdf
└── 2026_HYZ_OTR_Sablon_TR_*.docx
```

## Çalıştırma

```bash
cd competition_repo/TAKIM_BAGLANTI_ARAYUZU
cp config/example.env config/.env
# .env içine TEAM_NAME, PASSWORD, EVALUATION_SERVER_URL, SESSION_NAME doldur
pip install -r requirements.txt
python main.py
```

## detect() Metodu — Giriş/Çıkış

```python
def detect(self, prediction: FramePredictions, health_status: str) -> FramePredictions:
```

- `prediction.image_url`: İndirilen frame dosya yolu
- `prediction.gt_translation_x/y/z`: Ground truth (yalnızca health_status='1' güvenilir)
- `health_status`: `'1'` → GPS sağlıklı (GT'yi echo et) | `'0'` → GPS yok (VO çalıştır)

Her frame için yapılması gereken:
1. `prediction.add_detected_object(DetectedObject(cls, landing_status, x1, y1, x2, y2))`
2. `prediction.add_translation_object(DetectedTranslation(x, y, z))`

## Kamera Parametreleri (2025)

| | RGB | Termal |
|---|---|---|
| Çözünürlük | 4000×3000 | 640×512 |
| fx / fy | 2792.2 / 2795.2 | 731.8 / 732.0 |
| cx / cy | 1988.0 / 1562.2 | 319.2 / 251.2 |
| k1 / k2 | 0.0798 / -0.1867 | -0.3507 / 0.1137 |

## Sınıf ve Durum Sabitleri

```python
from .constants import classes, landing_statuses
# classes: {"Tasit": 0, "Insan": 1, "UAP": 2, "UAI": 3}
# landing_statuses: {"Inilebilir": 1, "Inilemez": 0, "InisAlaniDegil": -1}
```

## Sunucu Limitleri

- `send_prediction()`: Maksimum 80 frame/dakika
- `get_frames()` / `get_translations()`: Maksimum 5 istek/dakika
- Videolar: 7.5 FPS, ~2250 frame, `_images/{video_name}/` klasörüne indirilir

## Görev 2 — Visual Odometry Mimarisi

Başlangıç pozisyonu: (0.0, 0.0, 0.0)

- **health_status='1':** GT değerini doğrudan gönder + VO'yu bu frame ile kalibre et
- **health_status='0':** VO algoritması ile tahmin et

Yaklaşım: ORB feature matching → Essential Matrix → R/t decomposition → kümülatif pozisyon.
Scale belirsizliği için referans kare (30×30 cm checkerboard) kullanılabilir.
