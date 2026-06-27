import logging
import time
import requests
import os
import cv2
import numpy as np

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation

# ── Camera intrinsics (2025 calibration, RGB) ─────────────────────────────────
RGB_K = np.array([
    [2792.2, 0,      1988.0],
    [0,      2795.2, 1562.2],
    [0,      0,      1     ]
], dtype=np.float64)
RGB_D = np.array([0.0798, -0.1867, 0.0, 0.0], dtype=np.float64)

PROCESS_SCALE = 0.25

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

_GRID_ROWS, _GRID_COLS = 4, 6
_FEAT_PER_CELL         = 20
_SHI = dict(qualityLevel=0.01, minDistance=8, blockSize=7)


# =============================================================================
class VOInitializer:
    """
    Calibration from GPS=1 frames.

    Collects paired (GNSS_delta, pixel_delta) samples and at fit() computes
    per-axis scale factors (scale_x, scale_y) using rolling-median calibration.
    Per-axis is empirically more robust than a fixed R_inv matrix when the
    drone yaws continuously (camera heading changes during flight).
    """

    _WIN = 60   # rolling window size for median scale

    def __init__(self):
        self._sx_samples: list[float] = []
        self._sy_samples: list[float] = []
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.ready   = False

    def add_sample(self, gt_delta: np.ndarray, pixel_delta: np.ndarray):
        gt_dx, gt_dy = float(gt_delta[0]), float(gt_delta[1])
        tx,    ty    = float(pixel_delta[0]), float(pixel_delta[1])

        if abs(gt_dx) > 0.05 and abs(tx) > 1.0:
            sx = abs(gt_dx) / abs(tx)
            if 1e-4 < sx < 100.0:
                self._sx_samples.append(sx)
                if len(self._sx_samples) > self._WIN:
                    self._sx_samples.pop(0)

        if abs(gt_dy) > 0.05 and abs(ty) > 1.0:
            sy = abs(gt_dy) / abs(ty)
            if 1e-4 < sy < 100.0:
                self._sy_samples.append(sy)
                if len(self._sy_samples) > self._WIN:
                    self._sy_samples.pop(0)

        if self._sx_samples:
            self.scale_x = float(np.median(self._sx_samples))
        if self._sy_samples:
            self.scale_y = float(np.median(self._sy_samples))
        if self._sx_samples or self._sy_samples:
            self.ready = True

    def fit(self) -> bool:
        return self.ready


# =============================================================================
class FeatureTracker:
    """
    Grid-based Shi-Tomasi detection + LK tracking
    + 2-pass histogram outlier rejection (Paper 1).
    """

    MIN_FEATURES = 50

    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self._K    = K
        self._dist = dist
        self.prev_gray: np.ndarray | None = None
        self.prev_pts:  np.ndarray | None = None

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.undistort(gray, self._K, self._dist)
        return cv2.resize(gray, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)

    def _detect(self, gray: np.ndarray) -> np.ndarray | None:
        h, w = gray.shape
        rh, rw = h // _GRID_ROWS, w // _GRID_COLS
        pts = []
        for r in range(_GRID_ROWS):
            for c in range(_GRID_COLS):
                roi = gray[r*rh:(r+1)*rh, c*rw:(c+1)*rw]
                kps = cv2.goodFeaturesToTrack(roi, maxCorners=_FEAT_PER_CELL, **_SHI)
                if kps is not None:
                    kps[:, :, 0] += c * rw
                    kps[:, :, 1] += r * rh
                    pts.append(kps)
        return np.concatenate(pts) if pts else None

    @staticmethod
    def _histogram_reject(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        disps = curr - prev

        def modal_mask(vals: np.ndarray, tol: float = 5.0) -> np.ndarray:
            counts, edges = np.histogram(vals, bins=32)
            idx = int(np.argmax(counts))
            center = (edges[idx] + edges[idx + 1]) / 2.0
            return np.abs(vals - center) < tol

        mask1 = modal_mask(disps[:, 0]) & modal_mask(disps[:, 1])
        if mask1.sum() < 6:
            return mask1

        M, _ = cv2.estimateAffinePartial2D(
            prev[mask1], curr[mask1],
            method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if M is None:
            return mask1

        ones = np.ones((len(prev), 1), dtype=np.float32)
        pred = (M @ np.hstack([prev, ones]).T).T
        return np.linalg.norm(curr - pred, axis=1) < 1.5

    def update(self, bgr: np.ndarray) -> np.ndarray | None:
        gray  = self._preprocess(bgr)
        delta = None

        if (self.prev_gray is not None
                and self.prev_pts is not None
                and len(self.prev_pts) >= 10):

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **_LK
            )
            mask    = status.ravel().astype(bool)
            prev_ok = self.prev_pts[mask].reshape(-1, 2)
            curr_ok = curr_pts[mask].reshape(-1, 2)

            if len(prev_ok) >= 8:
                inliers = self._histogram_reject(prev_ok.astype(np.float32),
                                                 curr_ok.astype(np.float32))
                if inliers.sum() >= 6:
                    M, inl2 = cv2.estimateAffinePartial2D(
                        prev_ok[inliers], curr_ok[inliers],
                        method=cv2.RANSAC, ransacReprojThreshold=2.0
                    )
                    if M is not None and inl2 is not None and inl2.sum() >= 4:
                        delta = np.array([M[0, 2], M[1, 2]], dtype=np.float64)

                self.prev_pts = curr_ok.reshape(-1, 1, 2)
            else:
                self.prev_pts = None

        if self.prev_pts is None or len(self.prev_pts) < self.MIN_FEATURES:
            self.prev_pts = self._detect(gray)

        self.prev_gray = gray
        return delta

    def reset(self):
        self.prev_pts = None


# =============================================================================
class PoseEstimator:
    """Converts pixel displacement → world (dx, dy) using per-axis scale."""

    MAX_MOVE_M = 5.0

    def __init__(self, initializer: VOInitializer):
        self._init = initializer

    def estimate(self, pixel_delta: np.ndarray | None) -> np.ndarray:
        if pixel_delta is None or not self._init.ready:
            return np.zeros(3)

        dx = -float(pixel_delta[0]) * self._init.scale_x
        dy = -float(pixel_delta[1]) * self._init.scale_y

        if np.sqrt(dx**2 + dy**2) > self.MAX_MOVE_M:
            return np.zeros(3)

        return np.array([dx, dy, 0.0])


# =============================================================================
class PositionIntegrator:
    """Cumulative 3D position with a GPS=1-derived Z velocity model."""

    def __init__(self):
        self.pos  = np.zeros(3, dtype=np.float64)
        self._z_vel = 0.0
        self._z_buf: list[float] = []

    def update_z_model(self, gt_pos: np.ndarray, prev_gt: np.ndarray):
        self._z_buf.append(float(gt_pos[2] - prev_gt[2]))
        if len(self._z_buf) > 30:
            self._z_buf.pop(0)
        if len(self._z_buf) >= 5:
            self._z_vel = float(np.median(self._z_buf)) * 0.3

    def reset(self, gt_pos: np.ndarray):
        self.pos = gt_pos.copy()

    def step(self, disp: np.ndarray):
        self.pos[0] += disp[0]
        self.pos[1] += disp[1]
        self.pos[2] += self._z_vel


# =============================================================================
class ObjectDetectionModel:

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url

        self._initializer  = VOInitializer()
        self._tracker      = FeatureTracker(RGB_K, RGB_D)
        self._estimator    = PoseEstimator(self._initializer)
        self._integrator   = PositionIntegrator()

        self._prev_gt      = np.zeros(3, dtype=np.float64)
        self._prev_health  = '1'
        self._images_folder = None
        self.det_model     = None

    @staticmethod
    def download_image(img_url, images_folder, images_files,
                       retries=3, initial_wait_time=0.1):
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
                    logging.info(f'{img_url} - Downloaded in {time.perf_counter()-t1:.2f}s')
                    return
                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed {img_url} attempt {attempt+1}: {e}")
                    time.sleep(wait_time)
                    wait_time *= 2
            logging.error(f"Failed to download {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists, skipping.')

    def process(self, prediction, evaluation_server_url, health_status,
                images_folder, images_files):
        self.download_image(evaluation_server_url + "media" + prediction.image_url,
                            images_folder, images_files)
        self._images_folder = images_folder
        return self.detect(prediction, health_status)

    def _load_frame(self, prediction) -> np.ndarray | None:
        if self._images_folder is None:
            return None
        img_path = os.path.join(self._images_folder,
                                prediction.image_url.split("/")[-1])
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        return img

    def detect(self, prediction, health_status):
        frame = self._load_frame(prediction)

        if health_status == '1':
            gt_pos = np.array([
                float(prediction.gt_translation_x),
                float(prediction.gt_translation_y),
                float(prediction.gt_translation_z),
            ], dtype=np.float64)

            if frame is not None:
                pixel_delta = self._tracker.update(frame)
                if pixel_delta is not None:
                    self._initializer.add_sample(
                        gt_pos - self._prev_gt, pixel_delta
                    )

            self._integrator.update_z_model(gt_pos, self._prev_gt)

            if self._prev_health == '0':
                self._initializer.fit()
                self._integrator.reset(gt_pos)
                self._tracker.reset()

            self._prev_gt = gt_pos.copy()
            tx, ty, tz = gt_pos[0], gt_pos[1], gt_pos[2]

        else:
            if self._prev_health == '1':
                self._initializer.fit()
                self._integrator.reset(self._prev_gt)
                self._tracker.reset()

            if frame is not None:
                pixel_delta = self._tracker.update(frame)
                disp = self._estimator.estimate(pixel_delta)
                self._integrator.step(disp)

            tx = float(self._integrator.pos[0])
            ty = float(self._integrator.pos[1])
            tz = float(self._integrator.pos[2])

        self._prev_health = health_status
        prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
        return prediction
