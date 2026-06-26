"""
Standalone Visual Odometry test script.
Server gerektirmeden VO algoritmasını test eder.

Kullanım:
  python test_vo.py --smoke
  python test_vo.py --frames ../2025/THYZ_2025_Oturum_2-2 \\
                   --csv    ../2025/THYZ_2025_Oturum_2_Translation.csv
  python test_vo.py --frames ... --csv ... --plot
  python test_vo.py --frames ... --csv ... --save results.csv

CSV formatı (2025 yarışma):
  translation_x,translation_y,translation_z,frame_numbers
  0.004,−0.098,0.003,frame_000000
  ...
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# VO import (paket init gerektirmez)
# ---------------------------------------------------------------------------

def _import_vo():
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    sys.path.insert(0, os.path.dirname(__file__))
    import types
    pkg = types.ModuleType("src")
    pkg.__path__ = [src_dir]
    sys.modules.setdefault("src", pkg)
    for name in ["src.constants", "src.detected_object", "src.detected_translation"]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.classes = {}
            stub.landing_statuses = {}
            stub.DetectedObject = object
            stub.DetectedTranslation = object
            sys.modules[name] = stub
    from src.object_detection_model import VisualOdometry, RGB_CAMERA_MATRIX, RGB_DIST_COEFFS
    return VisualOdometry, RGB_CAMERA_MATRIX, RGB_DIST_COEFFS


# ---------------------------------------------------------------------------
# Yardımcı: GT CSV yükle
# ---------------------------------------------------------------------------

def load_gt_csv(csv_path: str) -> list[tuple[float, float, float]]:
    """
    2025 formatı: translation_x,translation_y,translation_z,frame_numbers
    Döner: sıralı [(x,y,z), ...] listesi (frame_000000 → indeks 0)
    """
    rows = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["frame_numbers"].strip()          # "frame_000042"
            idx = int(fname.split("_")[-1])               # 42
            rows[idx] = (float(row["translation_x"]),
                         float(row["translation_y"]),
                         float(row["translation_z"]))
    return [rows[i] for i in sorted(rows)]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    VisualOdometry, K, dist = _import_vo()
    vo = VisualOdometry(K, dist)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    assert vo.update(dummy).shape == (3,)
    assert vo.update(dummy).shape == (3,)
    print("Smoke test PASSED — VisualOdometry initializes and processes frames.")


# ---------------------------------------------------------------------------
# Ana test: frame klasörü + GT CSV
# ---------------------------------------------------------------------------

def run_frames(frames_dir: str, csv_path: str, health_split: float,
               plot: bool, save_path: str | None):

    VisualOdometry, K, dist = _import_vo()
    vo = VisualOdometry(K, dist)

    # Frame listesi
    exts = {".webp", ".jpg", ".jpeg", ".png"}
    frame_files = sorted(
        p for p in Path(frames_dir).iterdir() if p.suffix.lower() in exts
    )
    total = len(frame_files)
    if total == 0:
        print(f"HATA: {frames_dir} içinde frame bulunamadı.")
        sys.exit(1)

    # GT
    gt_list = load_gt_csv(csv_path) if csv_path else []
    has_gt = len(gt_list) > 0
    health_cutoff = int(total * health_split)

    print(f"\n{'='*60}")
    print(f"  Frame klasörü : {frames_dir}")
    print(f"  GT CSV        : {csv_path or '—'}")
    print(f"  Toplam frame  : {total}")
    print(f"  GPS=1 (health): ilk {health_cutoff} frame ({int(health_split*100)}%)")
    print(f"  GPS=0 (VO)    : kalan {total - health_cutoff} frame")
    print(f"{'='*60}\n")

    pred_positions: list[np.ndarray] = []
    errors: list[float] = []
    prev_gt = np.zeros(3)
    prev_health = "1"
    t_start = time.time()

    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            # webp fallback
            frame = cv2.imdecode(np.fromfile(str(fpath), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [UYARI] Okunamadı: {fpath.name}")
            pred_positions.append(pred_positions[-1].copy() if pred_positions else np.zeros(3))
            continue

        health = "1" if i < health_cutoff else "0"
        gt_pos = np.array(gt_list[i]) if has_gt and i < len(gt_list) else None

        if health == "1" and gt_pos is not None:
            vo.update(frame)
            vo.calibrate(gt_pos, prev_gt)
            if prev_health == "0":
                vo.reset_to(gt_pos)
            pos = gt_pos.copy()
            prev_gt = gt_pos.copy()
        else:
            # GPS yeni kesildi: VO'yu son bilinen GT konumuna sıfırla
            if prev_health == "1":
                vo.reset_to(prev_gt)
            pos = vo.update(frame)

        pred_positions.append(pos.copy())

        if gt_pos is not None and health == "0":
            err = float(np.linalg.norm(pos - gt_pos))
            errors.append(err)

        prev_health = health

        # İlerleme
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            rmse_now = float(np.sqrt(np.mean(np.array(errors)**2))) if errors else float("nan")
            print(f"  [{i+1:>4}/{total}]  pos=({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:6.2f})  "
                  f"FPS={fps:.1f}  RMSE={rmse_now:.4f} m")

    elapsed = time.time() - t_start

    # -----------------------------------------------------------------------
    # Sonuç özeti
    # -----------------------------------------------------------------------
    pred_arr = np.array(pred_positions)

    print(f"\n{'='*60}")
    print(f"  Tamamlandı: {total} frame  |  Süre: {elapsed:.1f}s  |  Ort. FPS: {total/elapsed:.1f}")
    if errors:
        rmse = float(np.sqrt(np.mean(np.array(errors)**2)))
        mae  = float(np.mean(np.abs(errors)))
        mx   = float(np.max(errors))
        print(f"  GPS=0 frame sayısı : {len(errors)}")
        print(f"  RMSE               : {rmse:.4f} m")
        print(f"  MAE                : {mae:.4f} m")
        print(f"  Max hata           : {mx:.4f} m")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Sonuçları CSV'ye kaydet
    # -----------------------------------------------------------------------
    if save_path:
        with open(save_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "pred_x", "pred_y", "pred_z",
                        "gt_x", "gt_y", "gt_z", "error_m", "health"])
            for i, pos in enumerate(pred_positions):
                gt = gt_list[i] if has_gt and i < len(gt_list) else ("", "", "")
                h = "1" if i < health_cutoff else "0"
                err = float(np.linalg.norm(pos - np.array(gt))) if has_gt and h == "0" else ""
                w.writerow([i, *pos.tolist(), *gt, err, h])
        print(f"  Sonuçlar kaydedildi → {save_path}")

    # -----------------------------------------------------------------------
    # Grafik
    # -----------------------------------------------------------------------
    if plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(14, 10))

            # 3D trajectory
            ax1 = fig.add_subplot(221, projection="3d")
            ax1.plot(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2],
                     "b-", linewidth=1, label="VO tahmini")
            ax1.plot(*pred_arr[0], "go", markersize=8, label="Başlangıç")
            ax1.plot(*pred_arr[-1], "rs", markersize=8, label="Bitiş")
            if has_gt:
                gt_arr = np.array(gt_list[:len(pred_positions)])
                ax1.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2],
                         "r--", linewidth=1, alpha=0.7, label="GT")
            ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
            ax1.set_title("3D Yörünge")
            ax1.legend(fontsize=8)

            # X-Y top view
            ax2 = fig.add_subplot(222)
            ax2.plot(pred_arr[:, 0], pred_arr[:, 1], "b-", linewidth=1, label="VO")
            if has_gt:
                ax2.plot(gt_arr[:, 0], gt_arr[:, 1], "r--", linewidth=1, alpha=0.7, label="GT")
            ax2.set_xlabel("X [m]"); ax2.set_ylabel("Y [m]")
            ax2.set_title("Üstten Görünüm (X-Y)")
            ax2.legend(fontsize=8); ax2.grid(True)

            # Z (yükseklik)
            ax3 = fig.add_subplot(223)
            ax3.plot(pred_arr[:, 2], "b-", linewidth=1, label="VO Z")
            if has_gt:
                ax3.plot([g[2] for g in gt_list[:len(pred_positions)]], "r--",
                         linewidth=1, alpha=0.7, label="GT Z")
            ax3.axvline(health_cutoff, color="orange", linestyle=":", label="GPS=0 başlangıcı")
            ax3.set_xlabel("Frame"); ax3.set_ylabel("Z [m]")
            ax3.set_title("Yükseklik (Z)")
            ax3.legend(fontsize=8); ax3.grid(True)

            # Hata grafiği
            ax4 = fig.add_subplot(224)
            if errors:
                ax4.plot(errors, "m-", linewidth=0.8, alpha=0.7)
                ax4.axhline(float(np.mean(errors)), color="red", linestyle="--",
                            label=f"Ort. {np.mean(errors):.3f} m")
                ax4.set_xlabel("GPS=0 Frame indeksi")
                ax4.set_ylabel("3D Hata [m]")
                ax4.set_title("Frame Bazlı Hata (GPS=0)")
                ax4.legend(fontsize=8); ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, "GT verisi yok", ha="center", va="center",
                         transform=ax4.transAxes)

            plt.suptitle("TUYGUN — Visual Odometry Test Sonuçları", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib kurulu değil — pip install matplotlib")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VO Test — server gerektirmez")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--frames", type=str, help="Frame klasörü (webp/jpg/png)")
    parser.add_argument("--csv", type=str, help="GT CSV (2025 yarışma formatı)")
    parser.add_argument("--health-split", type=float, default=0.2,
                        help="GPS=1 oranı (varsayılan 0.2 = ilk %%20)")
    parser.add_argument("--plot", action="store_true", help="Grafik göster")
    parser.add_argument("--save", type=str, default=None, help="Sonuçları CSV'ye kaydet")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if not args.frames:
        parser.print_help()
        sys.exit(1)

    run_frames(args.frames, args.csv, args.health_split, args.plot, args.save)


if __name__ == "__main__":
    main()
