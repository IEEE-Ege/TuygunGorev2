"""
Mock TEKNOFEST değerlendirme sunucusu.
Gerçek sunucu olmadan tam pipeline testi için.

Başlatma:
  python3 mock_server.py                        # varsayılan: sentetik video üret + sun
  python3 mock_server.py --video clip.mp4       # gerçek video dosyası sun

Ayrı terminalde:
  python3 main.py
"""

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
VIDEO_DIR = Path("./_mock_images/test_session/")
FRAMES_PER_SECOND = 7.5
TOKEN = "mock_token_123"

frames_data: list[dict] = []
translations_data: list[dict] = []
predictions_received: list[dict] = []

# ---------------------------------------------------------------------------
# Sentetik video üretici
# ---------------------------------------------------------------------------

def _generate_synthetic_video(out_path: Path, n_frames: int = 200):
    """
    Sabit yükseklikten aşağı bakan kamerayı simüle eder.
    Zemin: izgara desen. Kamera yavaş X-Y düzleminde ileri gider.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 540, 960
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vpath = str(out_path)
    writer = cv2.VideoWriter(vpath, fourcc, FRAMES_PER_SECOND, (w, h))

    for i in range(n_frames):
        # Kamera ofseti: yavaş X-Y hareketi
        cx = int(i * 2.5) % w
        cy = int(i * 1.5) % h
        img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        # Izgara çiz
        step = 80
        for x in range(0, w * 2, step):
            cv2.line(img, (x, 0), (x, h * 2), (80, 80, 80), 1)
        for y in range(0, h * 2, step):
            cv2.line(img, (0, y), (w * 2, y), (80, 80, 80), 1)
        # Bazı noktalar (feature için)
        rng = np.random.default_rng(42)
        for _ in range(60):
            px = int(rng.integers(0, w * 2))
            py = int(rng.integers(0, h * 2))
            cv2.circle(img, (px, py), 6, (0, 200, 255), -1)
        # Kırp — kamera hareketi simülasyonu
        crop = img[cy: cy + h, cx: cx + w]
        writer.write(crop)

    writer.release()
    print(f"[mock] Sentetik video oluşturuldu: {vpath}  ({n_frames} frame)")
    return vpath


def _build_frames_and_translations(video_path: str):
    """Video'yu frame frame okur, JPEG'e kaydeder, GT pozisyon üretir."""
    global frames_data, translations_data
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[mock] Video yüklendi: {total} frame")

    # Basit GT: düzgün dairesel yörünge (gerçekçi görünüm için)
    radius = 10.0  # metre
    alt = 30.0     # sabit yükseklik

    frames_data = []
    translations_data = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"frame_{idx:05d}.jpg"
        fpath = VIDEO_DIR / fname
        cv2.imwrite(str(fpath), frame)

        frames_data.append({
            "url": f"/frame_endpoint/{idx}/",
            "image_url": f"/test_session/{fname}",
            "video_name": "test_session",
        })

        # GT pozisyon: dairesel
        angle = (2 * math.pi * idx) / max(total, 1)
        x = radius * math.cos(angle) - radius   # başlangıç 0
        y = radius * math.sin(angle)
        z = alt

        # İlk %20 frame GPS sağlıklı, geri kalan GPS yok
        health = "1" if idx < total * 0.2 else "0"

        translations_data.append({
            "translation_x": str(round(x, 4)),
            "translation_y": str(round(y, 4)),
            "translation_z": str(round(z, 4)),
            "health_status": health,
        })
        idx += 1

    cap.release()
    print(f"[mock] {idx} frame hazırlandı. İlk %20 GPS=1, kalan GPS=0")


# ---------------------------------------------------------------------------
# Flask endpoint'leri  (gerçek sunucuyla birebir aynı)
# ---------------------------------------------------------------------------

@app.route("/auth/", methods=["POST"])
def login():
    return jsonify({"token": TOKEN}), 200


@app.route("/frames/", methods=["GET"])
def get_frames():
    if request.headers.get("Authorization") != f"Token {TOKEN}":
        return jsonify({"detail": "Unauthorized"}), 401
    return jsonify(frames_data), 200


@app.route("/translation/", methods=["GET"])
def get_translations():
    if request.headers.get("Authorization") != f"Token {TOKEN}":
        return jsonify({"detail": "Unauthorized"}), 401
    return jsonify(translations_data), 200


@app.route("/prediction/", methods=["POST"])
def send_prediction():
    if request.headers.get("Authorization") != f"Token {TOKEN}":
        return jsonify({"detail": "Unauthorized"}), 401
    data = request.get_json(force=True)
    predictions_received.append(data)
    n = len(predictions_received)
    if n % 50 == 0:
        _print_score()
    return jsonify({"status": "ok", "received": n}), 201


@app.route("/session/", methods=["GET"])
def session():
    return jsonify({"session_name": "test_session"}), 200


# Statik frame dosyası sun  (connection_handler media/ prefix'i ekliyor)
@app.route("/media/test_session/<path:filename>")
def serve_frame(filename):
    return send_from_directory(str(VIDEO_DIR), filename)


# ---------------------------------------------------------------------------
# Skor hesapla (terminale yaz)
# ---------------------------------------------------------------------------

def _print_score():
    if not predictions_received:
        return
    errors = []
    for pred in predictions_received:
        trans_list = pred.get("detected_translations", [])
        if not trans_list:
            continue
        t = trans_list[0]
        furl = pred.get("frame", "")
        # frame index'i URL'den çıkar
        try:
            idx = int(furl.strip("/").split("/")[-1])
        except (ValueError, IndexError):
            continue
        if idx >= len(translations_data):
            continue
        gt = translations_data[idx]
        dx = float(t["translation_x"]) - float(gt["translation_x"])
        dy = float(t["translation_y"]) - float(gt["translation_y"])
        dz = float(t["translation_z"]) - float(gt["translation_z"])
        errors.append(math.sqrt(dx**2 + dy**2 + dz**2))

    if errors:
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        print(f"[mock] Ara skor — {len(errors)} tahmin  |  RMSE: {rmse:.4f} m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Gerçek video dosyası (yoksa sentetik üretilir)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.video:
        video_path = args.video
    else:
        video_path = str(VIDEO_DIR.parent / "synthetic.mp4")
        if not Path(video_path).exists():
            _generate_synthetic_video(Path(video_path), n_frames=150)

    _build_frames_and_translations(video_path)

    # .env oluştur
    env_path = Path("./config/.env")
    env_path.parent.mkdir(exist_ok=True)
    env_path.write_text(
        f"TEAM_NAME=tuygun\n"
        f"PASSWORD=test123\n"
        f'EVALUATION_SERVER_URL="http://127.0.0.1:{args.port}/"\n'
        f"SESSION_NAME=test_session\n"
    )
    print(f"[mock] config/.env yazıldı → http://127.0.0.1:{args.port}/")
    print(f"[mock] Ayrı terminalde: source ../.venv/bin/activate && python3 main.py")
    print("-" * 60)

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
