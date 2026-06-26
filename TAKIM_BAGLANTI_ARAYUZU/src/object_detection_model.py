import logging
import time
import requests
import os
import cv2
import numpy as np
from collections import deque

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation


# RGB Camera intrinsics (2025 calibration)
RGB_CAMERA_MATRIX = np.array([
    [2792.2, 0,      1988.0],
    [0,      2795.2, 1562.2],
    [0,      0,      1     ]
], dtype=np.float64)

RGB_DIST_COEFFS = np.array([0.0798, -0.1867, 0.0, 0.0], dtype=np.float64)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)


class VisualOdometry:
    """
    Monocular VO for near-nadir UAV camera.

    XY : LK optical flow → affine transform → translation in pixels → metres
    Z  : altitude velocity model from GPS=1 history
    Scale (m/px): calibrated as GT_xy_delta / raw_pixel_delta — no feedback loop
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self.K = K
        self.dist = dist

        self.pos = np.zeros(3, dtype=np.float64)

        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

        # Per-axis scale: metres per pixel (calibrated from GT separately for X and Y)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._scale_x_samples: list[float] = []
        self._scale_y_samples: list[float] = []

        # Last raw pixel delta (set by update, read by calibrate)
        self._last_px_delta: np.ndarray | None = None

        # Z velocity model
        self._z_deltas: deque = deque(maxlen=30)
        self._z_velocity = 0.0

    # ------------------------------------------------------------------
    def _undistort(self, gray: np.ndarray) -> np.ndarray:
        return cv2.undistort(gray, self.K, self.dist)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)

    def _track(self, prev_gray, curr_gray, prev_pts):
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
        )
        mask = status.ravel().astype(bool)
        return prev_pts[mask], curr_pts[mask]

    def _affine_delta(self, src, dst) -> np.ndarray | None:
        """Fit partial affine (rotation+scale+translation); return tx,ty in pixels."""
        if len(src) < 8:
            return None
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if M is None or (inliers is not None and inliers.sum() < 6):
            return None
        return np.array([M[0, 2], M[1, 2]], dtype=np.float64)  # raw pixels

    # ------------------------------------------------------------------
    def calibrate(self, gt_pos: np.ndarray, prev_gt_pos: np.ndarray):
        """
        Update m/px scale and Z velocity from GT.
        Uses raw pixel delta stored by last update() — no feedback loop.
        """
        # Z velocity
        dz = gt_pos[2] - prev_gt_pos[2]
        self._z_deltas.append(dz)
        if len(self._z_deltas) >= 5:
            self._z_velocity = float(np.median(list(self._z_deltas)))

        # Per-axis scale: metres / pixel (X and Y calibrated independently)
        if self._last_px_delta is not None:
            px = self._last_px_delta
            gt_dx = gt_pos[0] - prev_gt_pos[0]
            gt_dy = gt_pos[1] - prev_gt_pos[1]

            if abs(gt_dx) > 0.05 and abs(px[0]) > 1.0:
                sx = abs(gt_dx) / abs(px[0])
                if 1e-6 < sx < 100.0:
                    self._scale_x_samples.append(sx)
                    if len(self._scale_x_samples) > 60:
                        self._scale_x_samples.pop(0)
                    self.scale_x = float(np.median(self._scale_x_samples))

            if abs(gt_dy) > 0.05 and abs(px[1]) > 1.0:
                sy = abs(gt_dy) / abs(px[1])
                if 1e-6 < sy < 100.0:
                    self._scale_y_samples.append(sy)
                    if len(self._scale_y_samples) > 60:
                        self._scale_y_samples.pop(0)
                    self.scale_y = float(np.median(self._scale_y_samples))

    def reset_to(self, gt_pos: np.ndarray):
        self.pos = gt_pos.copy()

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._undistort(gray)

        self._last_px_delta = None
        tracked_curr: np.ndarray | None = None

        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 10:
            prev_good, curr_good = self._track(self.prev_gray, gray, self.prev_pts)

            if len(prev_good) >= 8:
                delta_px = self._affine_delta(prev_good, curr_good)
                if delta_px is not None:
                    self._last_px_delta = delta_px.copy()
                    # Nadir camera: image shifts opposite to drone XY motion
                    self.pos[0] -= delta_px[0] * self.scale_x
                    self.pos[1] -= delta_px[1] * self.scale_y
                    self.pos[2] += self._z_velocity

                tracked_curr = curr_good.reshape(-1, 1, 2)

        # Refresh features when too few
        if tracked_curr is None or len(tracked_curr) < 50:
            self.prev_pts = self._detect_features(gray)
        else:
            self.prev_pts = tracked_curr

        self.prev_gray = gray
        return self.pos.copy()


class ObjectDetectionModel:

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url

        self.vo = VisualOdometry(RGB_CAMERA_MATRIX, RGB_DIST_COEFFS)
        self._prev_gt = np.zeros(3, dtype=np.float64)
        self._prev_health = '1'
        self._images_folder = None

        self.det_model = None

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]
        if image_name not in images_files:
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
                    response.raise_for_status()
                    with open(images_folder + image_name, 'wb') as f:
                        f.write(response.content)
                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Downloaded in {t2 - t1:.2f}s')
                    return
                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed {img_url} attempt {attempt+1}: {e}")
                    time.sleep(wait_time)
                    wait_time *= 2
            logging.error(f"Failed to download {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists, skipping.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files)
        self._images_folder = images_folder
        return self.detect(prediction, health_status)

    def _run_object_detection(self, img_path: str, prediction):
        if self.det_model is None:
            return

    def _load_frame(self, prediction) -> np.ndarray | None:
        if self._images_folder is None:
            return None
        image_name = prediction.image_url.split("/")[-1]
        img_path = os.path.join(self._images_folder, image_name)
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def detect(self, prediction, health_status):
        img_path = os.path.join(
            self._images_folder or "",
            prediction.image_url.split("/")[-1]
        )

        self._run_object_detection(img_path, prediction)

        frame = self._load_frame(prediction)

        if health_status == '1':
            gt_pos = np.array([
                float(prediction.gt_translation_x),
                float(prediction.gt_translation_y),
                float(prediction.gt_translation_z),
            ], dtype=np.float64)

            if frame is not None:
                self.vo.update(frame)
                self.vo.calibrate(gt_pos, self._prev_gt)
                if self._prev_health == '0':
                    self.vo.reset_to(gt_pos)

            self._prev_gt = gt_pos.copy()
            tx, ty, tz = gt_pos[0], gt_pos[1], gt_pos[2]

        else:
            # GPS yeni kesildi: VO'yu son bilinen GT konumuna sıfırla
            if self._prev_health == '1':
                self.vo.reset_to(self._prev_gt)
            if frame is not None:
                pos = self.vo.update(frame)
            else:
                pos = self.vo.pos.copy()
            tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

        self._prev_health = health_status
        prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
        return prediction
