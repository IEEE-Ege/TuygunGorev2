"""
VO test script — server gerektirmez.
Kullanım:
  python3 test_vo.py --frames ../2025/THYZ_2025_Oturum_2-2 \
                     --csv    ../2025/THYZ_2025_Oturum_2_Translation.csv
"""
import argparse, csv, os, sys, time, types
from pathlib import Path
import cv2
import numpy as np


def _bootstrap():
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    sys.path.insert(0, os.path.dirname(__file__))
    pkg = types.ModuleType("src")
    pkg.__path__ = [src_dir]
    sys.modules.setdefault("src", pkg)
    for name in ["src.constants", "src.detected_object", "src.detected_translation"]:
        if name not in sys.modules:
            s = types.ModuleType(name)
            s.classes = {}; s.landing_statuses = {}
            s.DetectedObject = object
            class _DT:
                def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
            s.DetectedTranslation = _DT
            sys.modules[name] = s
    from src.object_detection_model import (
        VOInitializer, FeatureTracker, PoseEstimator,
        PositionIntegrator, RGB_K, RGB_D
    )
    return VOInitializer, FeatureTracker, PoseEstimator, PositionIntegrator, RGB_K, RGB_D


def load_gt(csv_path):
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_numbers"].strip().split("_")[-1])
            rows[idx] = (float(row["translation_x"]),
                         float(row["translation_y"]),
                         float(row["translation_z"]))
    return [rows[i] for i in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--csv",    required=True)
    ap.add_argument("--split",  type=float, default=0.2, help="GPS=1 oranı")
    ap.add_argument("--save",   default=None)
    args = ap.parse_args()

    VOInitializer, FeatureTracker, PoseEstimator, PositionIntegrator, RGB_K, RGB_D = _bootstrap()

    initializer  = VOInitializer()
    tracker      = FeatureTracker(RGB_K, RGB_D)
    estimator    = PoseEstimator(initializer)
    integrator   = PositionIntegrator()

    exts = {".webp", ".jpg", ".jpeg", ".png"}
    frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in exts)
    gt     = load_gt(args.csv)
    total  = len(frames)
    cut    = int(total * args.split)

    print(f"\nToplam frame: {total}  |  GPS=1: {cut}  |  GPS=0: {total-cut}\n")

    preds, errors = [], []
    prev_gt      = np.zeros(3)
    prev_health  = "1"
    t0 = time.time()

    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        if frame is None:
            frame = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            preds.append(preds[-1].copy() if preds else np.zeros(3))
            continue

        health = "1" if i < cut else "0"
        gt_pos = np.array(gt[i]) if i < len(gt) else None

        if health == "1":
            pixel_delta = tracker.update(frame)
            if pixel_delta is not None and gt_pos is not None:
                initializer.add_sample(gt_pos - prev_gt, pixel_delta)
            if gt_pos is not None:
                integrator.update_z_model(gt_pos, prev_gt)
            if prev_health == "0":
                initializer.fit()
                if gt_pos is not None:
                    integrator.reset(gt_pos)
                tracker.reset()
            pos = gt_pos.copy() if gt_pos is not None else prev_gt.copy()
            if gt_pos is not None:
                prev_gt = gt_pos.copy()
        else:
            if prev_health == "1":
                initializer.fit()
                integrator.reset(prev_gt)
                tracker.reset()
            pixel_delta = tracker.update(frame)
            disp = estimator.estimate(pixel_delta)
            integrator.step(disp)
            pos = integrator.pos.copy()

        preds.append(pos.copy())
        if health == "0" and gt_pos is not None:
            errors.append(float(np.linalg.norm(pos - gt_pos)))

        prev_health = health

        if (i + 1) % 100 == 0 or (i + 1) == total:
            rmse = float(np.sqrt(np.mean(np.array(errors)**2))) if errors else float("nan")
            fps  = (i + 1) / (time.time() - t0)
            print(f"  [{i+1:4d}/{total}]  pos=({pos[0]:7.2f},{pos[1]:7.2f},{pos[2]:6.2f})"
                  f"  RMSE={rmse:.2f}m  FPS={fps:.1f}")

    print(f"\n{'='*55}")
    if errors:
        arr  = np.array(errors)
        rmse = float(np.sqrt(np.mean(arr**2)))
        print(f"  RMSE : {rmse:.4f} m")
        print(f"  MAE  : {np.mean(arr):.4f} m")
        print(f"  Max  : {np.max(arr):.4f} m")
        print(f"  N    : {len(errors)}")

    # Kalibrasyon özeti
    print(f"\n  VOInitializer:")
    print(f"    scale_x = {initializer.scale_x:.6f} m/px  (n={len(initializer._sx_samples)})")
    print(f"    scale_y = {initializer.scale_y:.6f} m/px  (n={len(initializer._sy_samples)})")
    print(f"{'='*55}\n")

    if args.save:
        with open(args.save, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame","pred_x","pred_y","pred_z","gt_x","gt_y","gt_z","err","health"])
            for i, pos in enumerate(preds):
                h  = "1" if i < cut else "0"
                gp = gt[i] if i < len(gt) else ("","","")
                e  = float(np.linalg.norm(pos - np.array(gp))) if h == "0" and i < len(gt) else ""
                w.writerow([i, *pos.tolist(), *gp, e, h])
        print(f"  Kaydedildi → {args.save}")


if __name__ == "__main__":
    main()
