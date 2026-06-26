"""
Standalone Visual Odometry test script.
Server gerektirmeden VO algoritmasını test eder.

Kullanım:
  python test_vo.py --smoke                          # import + init testi
  python test_vo.py --video clip.mp4                 # sadece VO çalıştır
  python test_vo.py --video clip.mp4 --gt gt.csv     # RMSE hesapla
  python test_vo.py --video clip.mp4 --gt gt.csv --plot  # 3D trajectory çiz

GT CSV formatı (başlık satırı zorunlu):
  frame_id,x,y,z,health
  0,0.0,0.0,0.0,1
  1,0.12,0.05,0.01,1
  ...
"""

import argparse
import csv
import sys
import os

import cv2
import numpy as np


def _import_vo():
    """Import VisualOdometry directly (no package init needed)."""
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    sys.path.insert(0, os.path.dirname(__file__))

    # Minimal stub for package-level imports
    import types
    pkg = types.ModuleType("src")
    pkg.__path__ = [src_dir]
    sys.modules.setdefault("src", pkg)

    # Stub out unused imports inside object_detection_model
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


def smoke_test():
    VisualOdometry, K, dist = _import_vo()
    vo = VisualOdometry(K, dist)
    # Feed two black frames — should not crash
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    pos1 = vo.update(dummy)
    pos2 = vo.update(dummy)
    assert pos1.shape == (3,)
    assert pos2.shape == (3,)
    print("Smoke test PASSED — VisualOdometry initializes and processes frames.")


def run_video(video_path: str, gt_path: str | None, plot: bool):
    VisualOdometry, K, dist = _import_vo()
    vo = VisualOdometry(K, dist)

    # Load GT if provided
    gt_data: dict[int, tuple[np.ndarray, str]] = {}
    if gt_path:
        with open(gt_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = int(row["frame_id"])
                pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
                health = row.get("health", "0").strip()
                gt_data[fid] = (pos, health)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video '{video_path}'")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  |  Frames: {total}")

    pred_positions: list[np.ndarray] = []
    gt_positions: list[np.ndarray] = []
    prev_gt = np.zeros(3, dtype=np.float64)
    prev_health = "1"
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        health = "0"
        gt_pos = None
        if frame_id in gt_data:
            gt_pos, health = gt_data[frame_id]

        if health == "1" and gt_pos is not None:
            vo.update(frame)
            vo.calibrate_scale(gt_pos, prev_gt)
            if prev_health == "0":
                vo.reset_to(gt_pos)
            pos = gt_pos.copy()
            prev_gt = gt_pos.copy()
        else:
            pos = vo.update(frame)

        pred_positions.append(pos.copy())
        if gt_pos is not None:
            gt_positions.append((frame_id, gt_pos.copy()))

        prev_health = health
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Frame {frame_id}/{total}  pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    cap.release()
    print(f"\nProcessed {frame_id} frames.")

    # RMSE
    if gt_positions:
        errors = []
        for fid, gt in gt_positions:
            if fid < len(pred_positions):
                err = np.linalg.norm(pred_positions[fid] - gt)
                errors.append(err)
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        print(f"RMSE (3D Euclidean): {rmse:.4f} m  over {len(errors)} GT frames")

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            pred_arr = np.array(pred_positions)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2], "b-", label="VO estimate")
            if gt_positions:
                gt_arr = np.array([g for _, g in gt_positions])
                ax.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], "r--", label="Ground Truth")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.legend()
            ax.set_title("Visual Odometry Trajectory")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot.")


def main():
    parser = argparse.ArgumentParser(description="Test Visual Odometry without server")
    parser.add_argument("--smoke", action="store_true", help="Quick import + init test")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--gt", type=str, help="Path to ground-truth CSV")
    parser.add_argument("--plot", action="store_true", help="Show 3D trajectory plot")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if not args.video:
        parser.print_help()
        sys.exit(1)

    run_video(args.video, args.gt, args.plot)


if __name__ == "__main__":
    main()
