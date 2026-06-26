import logging
import time
import requests
import os
import cv2
import numpy as np

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

MIN_MATCHES = 30  # Minimum good matches to trust motion estimate


class VisualOdometry:
    """Monocular VO via ORB feature matching + Essential Matrix decomposition."""

    def __init__(self, K: np.ndarray, dist: np.ndarray):
        self.K = K
        self.dist = dist
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Cumulative pose in world frame
        self.pos = np.zeros(3, dtype=np.float64)
        self.R_total = np.eye(3, dtype=np.float64)

        # Previous frame state
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        # Scale calibration
        self.scale = 1.0
        self._scale_samples: list[float] = []
        self._prev_vo_pos = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    def _undistort(self, gray: np.ndarray) -> np.ndarray:
        return cv2.undistort(gray, self.K, self.dist)

    def _detect(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def _match(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            return []
        raw = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        return good

    def _estimate_motion(self, kp1, kp2, matches):
        """Returns R, t (unit vector) or (None, None) on failure."""
        if len(matches) < MIN_MATCHES:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t  # t is unit vector in camera1 coords

    # ------------------------------------------------------------------
    def calibrate_scale(self, gt_pos: np.ndarray, prev_gt_pos: np.ndarray):
        """Update scale using consecutive GT frame distance vs VO delta."""
        gt_delta = np.linalg.norm(gt_pos - prev_gt_pos)
        vo_delta = np.linalg.norm(self.pos - self._prev_vo_pos)
        if gt_delta > 0.05 and vo_delta > 1e-4:
            self._scale_samples.append(gt_delta / vo_delta)
            if len(self._scale_samples) > 50:
                self._scale_samples.pop(0)
            self.scale = float(np.median(self._scale_samples))

    def reset_to(self, gt_pos: np.ndarray):
        """Hard-reset cumulative position to ground truth."""
        self.pos = gt_pos.copy()

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process a new frame; return current cumulative position."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._undistort(gray)
        kp, des = self._detect(gray)

        if self.prev_gray is not None and self.prev_kp is not None:
            matches = self._match(self.prev_des, des)
            R, t = self._estimate_motion(self.prev_kp, kp, matches)

            if R is not None and t is not None:
                self._prev_vo_pos = self.pos.copy()
                # t is in camera1 frame → rotate to world frame, then scale
                self.pos += self.scale * (self.R_total @ t).flatten()
                self.R_total = self.R_total @ R

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return self.pos.copy()


class ObjectDetectionModel:

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url

        self.vo = VisualOdometry(RGB_CAMERA_MATRIX, RGB_DIST_COEFFS)
        self._prev_gt = np.zeros(3, dtype=np.float64)
        self._prev_health = '1'   # track health transitions
        self._images_folder = None

        # Object detection model placeholder
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

    # ------------------------------------------------------------------
    def _run_object_detection(self, img_path: str, prediction):
        """
        Nesne tespiti entegrasyon noktası (Görev 1).
        YOLO veya başka model buraya eklenir.
        """
        if self.det_model is None:
            return
        # Örnek (ultralytics YOLO):
        # results = self.det_model(img_path)
        # for box in results[0].boxes:
        #     cls_id = int(box.cls)
        #     x1, y1, x2, y2 = box.xyxy[0].tolist()
        #     prediction.add_detected_object(
        #         DetectedObject(cls_id, landing_statuses["InisAlaniDegil"], x1, y1, x2, y2)
        #     )

    def _load_frame(self, prediction) -> np.ndarray | None:
        if self._images_folder is None:
            return None
        image_name = prediction.image_url.split("/")[-1]
        img_path = os.path.join(self._images_folder, image_name)
        if not os.path.exists(img_path):
            return None
        return cv2.imread(img_path)

    # ------------------------------------------------------------------
    def detect(self, prediction, health_status):
        image_name = prediction.image_url.split("/")[-1]
        img_path = os.path.join(self._images_folder or "", image_name)

        # --- Görev 1: Nesne tespiti ---
        self._run_object_detection(img_path, prediction)

        # --- Görev 2: Pozisyon kestirimi ---
        frame = self._load_frame(prediction)

        if health_status == '1':
            gt_pos = np.array([
                float(prediction.gt_translation_x),
                float(prediction.gt_translation_y),
                float(prediction.gt_translation_z),
            ], dtype=np.float64)

            if frame is not None:
                self.vo.update(frame)
                self.vo.calibrate_scale(gt_pos, self._prev_gt)
                # Reset yalnızca GPS geri geldiğinde (0→1 geçişi)
                if self._prev_health == '0':
                    self.vo.reset_to(gt_pos)

            self._prev_gt = gt_pos.copy()
            tx, ty, tz = gt_pos[0], gt_pos[1], gt_pos[2]

        else:  # health_status == '0'
            if frame is not None:
                pos = self.vo.update(frame)
            else:
                pos = self.vo.pos.copy()
            tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

        self._prev_health = health_status
        prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
        return prediction
