"""
Mock TEKNOFEST değerlendirme sunucusu.
Gerçek sunucu olmadan tam pipeline testi için.

Kullanım:
  # Gerçek frame klasörü + CSV ile:
  python3 mock_server.py --frames ../2025/THYZ_2025_Oturum_2-2 --csv ../2025/THYZ_2025_Oturum_2_Translation.csv

  # Sentetik (frame/csv yoksa):
  python3 mock_server.py

Ayrı terminalde:
  python3 main.py
"""

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

TOKEN = "mock_token_123"
SESSION_NAME = "test_session"

frames_data: list[dict] = []
translations_data: list[dict] = []
predictions_received: list[dict] = []
_frames_dir: Path = Path(".")


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_from_folder_and_csv(frames_dir: Path, csv_path: Path, health_split: float = 0.2):
    """
    frames_dir : frame_000000.webp / .jpg / .png içeren klasör
    csv_path   : translation_x,translation_y,translation_z,frame_numbers
    health_split: ilk bu oran GPS=1, geri kalan GPS=0
    """
    global frames_data, translations_data, _frames_dir
    _frames_dir = frames_dir

    # CSV oku
    gt: dict[str, tuple[float, float, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["frame_numbers"].strip()
            gt[fname] = (float(row["translation_x"]),
                         float(row["translation_y"]),
                         float(row["translation_z"]))

    # Frame dosyalarını sırala
    exts = {".webp", ".jpg", ".jpeg", ".png"}
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    total = len(files)
    print(f"[mock] {total} frame, {len(gt)} GT satırı bulundu.")

    frames_data = []
    translations_data = []

    for i, fpath in enumerate(files):
        stem = fpath.stem  # e.g. "frame_000000"
        fname_key = stem   # CSV'deki frame_numbers değeriyle eşleş

        frames_data.append({
            "url": f"/frame_endpoint/{i}/",
            "image_url": f"/{SESSION_NAME}/{fpath.name}",
            "video_name": SESSION_NAME,
        })

        if fname_key in gt:
            x, y, z = gt[fname_key]
        else:
            x, y, z = 0.0, 0.0, 0.0

        health = "1" if i < total * health_split else "0"
        translations_data.append({
            "translation_x": str(x),
            "translation_y": str(y),
            "translation_z": str(z),
            "health_status": health,
        })

    print(f"[mock] İlk %{int(health_split*100)} GPS=1 ({int(total*health_split)} frame), "
          f"kalan GPS=0 ({total - int(total*health_split)} frame)")


def generate_synthetic(n_frames: int = 150):
    """Sentetik frame + GT üretir (gerçek veri yoksa)."""
    global frames_data, translations_data, _frames_dir
    out_dir = Path("./_mock_images/test_session")
    out_dir.mkdir(parents=True, exist_ok=True)
    _frames_dir = out_dir

    h, w = 540, 960
    radius = 10.0
    alt = 30.0
    rng = np.random.default_rng(42)

    frames_data = []
    translations_data = []

    for i in range(n_frames):
        # Izgara görüntü
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cx_off = int(i * 2.5) % 80
        cy_off = int(i * 1.5) % 80
        for x in range(-cx_off, w, 80):
            cv2.line(img, (x, 0), (x, h), (80, 80, 80), 1)
        for y in range(-cy_off, h, 80):
            cv2.line(img, (0, y), (w, y), (80, 80, 80), 1)
        for _ in range(40):
            px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(img, (px, py), 5, (0, 200, 255), -1)

        fname = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), img)

        angle = (2 * math.pi * i) / n_frames
        x_gt = radius * math.cos(angle) - radius
        y_gt = radius * math.sin(angle)

        frames_data.append({
            "url": f"/frame_endpoint/{i}/",
            "image_url": f"/{SESSION_NAME}/{fname}",
            "video_name": SESSION_NAME,
        })
        health = "1" if i < n_frames * 0.2 else "0"
        translations_data.append({
            "translation_x": str(round(x_gt, 4)),
            "translation_y": str(round(y_gt, 4)),
            "translation_z": str(round(alt, 4)),
            "health_status": health,
        })

    print(f"[mock] Sentetik {n_frames} frame oluşturuldu → {out_dir}")


# ---------------------------------------------------------------------------
# Flask endpoint'leri
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
    if n % 100 == 0 or n == len(frames_data):
        _print_score()
    return jsonify({"status": "ok", "received": n}), 201


@app.route("/session/", methods=["GET"])
def session():
    return jsonify({"session_name": SESSION_NAME}), 200


@app.route(f"/media/{SESSION_NAME}/<path:filename>")
def serve_frame(filename):
    return send_from_directory(str(_frames_dir), filename)


# ---------------------------------------------------------------------------
# Skor
# ---------------------------------------------------------------------------

def _print_score():
    errors = []
    for pred in predictions_received:
        trans_list = pred.get("detected_translations", [])
        if not trans_list:
            continue
        t = trans_list[0]
        furl = pred.get("frame", "")
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
        print(f"[mock] {len(errors)}/{len(frames_data)} tahmin  |  RMSE: {rmse:.4f} m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, default=None, help="Frame klasörü (webp/jpg/png)")
    parser.add_argument("--csv", type=str, default=None, help="GT CSV dosyası")
    parser.add_argument("--health-split", type=float, default=0.2,
                        help="İlk bu oran GPS=1, kalan GPS=0 (varsayılan 0.2)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.frames and args.csv:
        load_from_folder_and_csv(Path(args.frames), Path(args.csv), args.health_split)
    else:
        print("[mock] --frames/--csv verilmedi, sentetik veri kullanılıyor.")
        generate_synthetic()

    # .env oluştur
    env_path = Path("./config/.env")
    env_path.parent.mkdir(exist_ok=True)
    env_path.write_text(
        f"TEAM_NAME=tuygun\n"
        f"PASSWORD=test123\n"
        f'EVALUATION_SERVER_URL=http://127.0.0.1:{args.port}/\n'
        f"SESSION_NAME={SESSION_NAME}\n"
    )
    print(f"[mock] config/.env yazıldı")
    print(f"[mock] Sunucu: http://127.0.0.1:{args.port}/")
    print(f"[mock] Ayrı terminalde çalıştır: python3 main.py")
    print("-" * 60)

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
