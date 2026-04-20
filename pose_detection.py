from __future__ import annotations
from dataclasses import dataclass
from math import exp
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    import cv2
    import mediapipe as mp
except ImportError:  # Allows importing this module in environments without CV deps.
    cv2 = None
    mp = None
@dataclass
class PostureFeatures:
    torso_verticality: float
    hip_knee_angle: float
    head_to_floor_distance: float
    bounding_box_ratio: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.torso_verticality,
                self.hip_knee_angle,
                self.head_to_floor_distance,
                self.bounding_box_ratio,
            ],
            dtype=np.float32,
        )

def _clip01(value: float)->float:
    return float(max(0.0, min(1.0, value)))

def _sigmoid(z: float)->float:
    return 1.0 / (1.0 + exp(-z))

def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray)->float:
    """Compute angle ABC in degrees."""
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    return angle

class PostureRiskDetector:
    """Posture detection and risk scoring using MediaPipe landmarks + logistic model.
    Features used:
    1) torso_verticality      : |shoulder_y - hip_y| (smaller means likely horizontal)
    2) hip_knee_angle         : angle at hip between shoulder-hip-knee
    3) head_to_floor_distance : nose_y (larger means closer to floor in normalized frame)
    4) bounding_box_ratio     : bbox_height / bbox_width (smaller means horizontal spread)"""

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self,threshold: float = 0.60,coefficients: Optional[List[float]] = None,intercept: float = 0.0,min_visibility: float = 0.5,pose_model_path: Optional[str] = None,)->None:
        """coefficients order:
        [torso_verticality, hip_knee_angle, head_to_floor_distance, bounding_box_ratio]

        Default coefficients are hand-picked priors and should be replaced by
        fitted values once you collect labeled posture data."""
        self.threshold = threshold
        self.intercept = intercept
        self.min_visibility = min_visibility
        self.coefficients = np.array(
            coefficients if coefficients is not None else [-3.2, -0.02, 2.4, -2.0],
            dtype=np.float32,
        )

        if self.coefficients.shape[0] != 4:
            raise ValueError("coefficients must contain exactly 4 values.")

        self._pose = None
        self._landmarker = None
        self._backend = None

        if mp is not None and hasattr(mp, "solutions"):
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            self._backend = "solutions"
        elif mp is not None:
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions
            except ImportError as exc:
                raise ImportError(
                    "mediapipe is installed but no supported pose API is available."
                ) from exc

            if not pose_model_path:
                raise ValueError(
                    "This mediapipe build requires the Tasks API model file. "
                    "Provide pose_model_path to PostureRiskDetector(...) or pass "
                    "--pose-model in CLI."
                )

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._backend = "tasks"

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _landmark_to_xy(
        self, landmarks: List, idx: int
    ) -> Tuple[np.ndarray, float]:
        lm = landmarks[idx]
        point = np.array([lm.x, lm.y], dtype=np.float32)
        visibility = float(getattr(lm, "visibility", 1.0))
        return point, visibility

    def extract_features_from_landmarks(self, landmarks: List) -> PostureFeatures:
        l_sh, v1 = self._landmark_to_xy(landmarks, self.LEFT_SHOULDER)
        r_sh, v2 = self._landmark_to_xy(landmarks, self.RIGHT_SHOULDER)
        l_hip, v3 = self._landmark_to_xy(landmarks, self.LEFT_HIP)
        r_hip, v4 = self._landmark_to_xy(landmarks, self.RIGHT_HIP)
        l_knee, v5 = self._landmark_to_xy(landmarks, self.LEFT_KNEE)
        r_knee, v6 = self._landmark_to_xy(landmarks, self.RIGHT_KNEE)
        nose, v7 = self._landmark_to_xy(landmarks, self.NOSE)

        visibilities = [v1, v2, v3, v4, v5, v6, v7]
        if min(visibilities) < self.min_visibility:
            raise ValueError("Critical landmarks are not visible enough.")

        shoulder_center = (l_sh + r_sh) / 2.0
        hip_center = (l_hip + r_hip) / 2.0
        knee_center = (l_knee + r_knee) / 2.0

        torso_verticality = abs(float(shoulder_center[1] - hip_center[1]))
        hip_knee_angle = _angle_degrees(shoulder_center, hip_center, knee_center)
        head_to_floor_distance = _clip01(float(nose[1]))

        all_points = np.array(
            [l_sh, r_sh, l_hip, r_hip, l_knee, r_knee, nose], dtype=np.float32
        )
        min_x = float(np.min(all_points[:, 0]))
        max_x = float(np.max(all_points[:, 0]))
        min_y = float(np.min(all_points[:, 1]))
        max_y = float(np.max(all_points[:, 1]))
        box_h = max(max_y - min_y, 1e-6)
        box_w = max(max_x - min_x, 1e-6)
        bounding_box_ratio = box_h / box_w

        return PostureFeatures(
            torso_verticality=torso_verticality,
            hip_knee_angle=hip_knee_angle,
            head_to_floor_distance=head_to_floor_distance,
            bounding_box_ratio=bounding_box_ratio,
        )

    def posture_label(self, features: PostureFeatures) -> str:
        angle = features.hip_knee_angle
        ratio = features.bounding_box_ratio
        torso = features.torso_verticality

        if torso < 0.08 and ratio < 1.0:
            if angle < 45:
                return "fetal_like"
            if angle > 150:
                return "lying_flat"
            return "lying_side"
        if 70 <= angle <= 115:
            return "sitting"
        return "standing_or_other"

    def risk_score(self, features: PostureFeatures) -> float:
        x = features.as_array()
        z = float(self.intercept + np.dot(self.coefficients, x))
        return _clip01(_sigmoid(z))

    def analyze_landmarks(self, landmarks: List) -> Dict:
        features = self.extract_features_from_landmarks(landmarks)
        score = self.risk_score(features)
        label = self.posture_label(features)
        return {
            "posture_label": label,
            "posture_score": score,
            "is_high_risk": score >= self.threshold,
            "features": {
                "torso_verticality": round(features.torso_verticality, 4),
                "hip_knee_angle": round(features.hip_knee_angle, 2),
                "head_to_floor_distance": round(features.head_to_floor_distance, 4),
                "bounding_box_ratio": round(features.bounding_box_ratio, 4),
            },
        }

    def _tasks_landmarks_to_points(self, landmarks: List[Any]) -> List[Any]:
        class _Point:
            def __init__(self, x: float, y: float, visibility: float) -> None:
                self.x = x
                self.y = y
                self.visibility = visibility

        return [
            _Point(
                float(lm.x),
                float(lm.y),
                float(getattr(lm, "visibility", 1.0)),
            )
            for lm in landmarks
        ]

    def analyze_image(self, image_bgr: np.ndarray) -> Dict:
        if cv2 is None:
            raise ImportError(
                "opencv-python is required. Install with: pip install opencv-python"
            )
        if mp is None:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            )

        if self._backend == "solutions":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                return {
                    "posture_label": "no_person_detected",
                    "posture_score": 0.0,
                    "is_high_risk": False,
                    "features": None,
                }

            landmarks = result.pose_landmarks.landmark
            return self.analyze_landmarks(landmarks)

        if self._backend == "tasks":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            if not result.pose_landmarks:
                return {
                    "posture_label": "no_person_detected",
                    "posture_score": 0.0,
                    "is_high_risk": False,
                    "features": None,
                }

            landmarks = self._tasks_landmarks_to_points(result.pose_landmarks[0])
            return self.analyze_landmarks(landmarks)

        raise RuntimeError(
            "Pose detector backend is not initialized. "
            "Check mediapipe installation and constructor parameters."
        )

    def analyze_image_path(self, image_path: str) -> Dict:
        if cv2 is None:
            raise ImportError("opencv-python is required to read image files.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at path: {image_path}")
        return self.analyze_image(image)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Posture risk detection demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.7, help="Risk threshold")
    parser.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Path to MediaPipe .task model (needed for Tasks API builds).",
    )
    args = parser.parse_args()

    detector = PostureRiskDetector(
        threshold=args.threshold,
        pose_model_path=args.pose_model,
    )
    try:
        output = detector.analyze_image_path(args.image)
        print(json.dumps(output, indent=2))
    finally:
        detector.close()
