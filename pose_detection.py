from __future__ import annotations
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve
import numpy as np
try:
    import cv2
    import mediapipe as mp
except ImportError:  
    cv2 = None
    mp = None

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

class PostureFeatures:
    torso_verticality: float
    hip_knee_angle: float
    head_to_floor_distance: float
    bounding_box_ratio: float

    def as_array(self) -> np.ndarray:
        return np.array([self.torso_verticality,self.hip_knee_angle,self.head_to_floor_distance,self.bounding_box_ratio,],dtype=np.float32)

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
    DEFAULT_TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

    def __init__(self,threshold: float = 0.20,coefficients: Optional[List[float]]=None,intercept: float = -1.4291609262079459,min_visibility: float = 0.5,pose_model_path: Optional[str] = None,auto_download_model: bool = True,)->None:
        """coefficients order:
        [torso_verticality, hip_knee_angle, head_to_floor_distance, bounding_box_ratio]"""
        
        self.threshold = threshold
        self.intercept = intercept
        self.min_visibility = min_visibility
        self.coefficients = np.array(
            coefficients if coefficients is not None else [-1.5334397692302808,0.032520078156034415,1.4150597448356768,-8.23687911238297],
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
                pose_model_path = str((Path.cwd() / "pose_landmarker_lite.task").resolve())
            model_path = Path(pose_model_path).resolve()
            if not model_path.exists():
                if not auto_download_model:
                    raise ValueError(
                        f"Pose model file not found: {model_path}. "
                        "Set auto_download_model=True or pass --pose-model."
                    )
                self._download_pose_model(model_path)

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._backend = "tasks"

    def _download_pose_model(cls, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading pose model to: {model_path}")
            urlretrieve(cls.DEFAULT_TASK_MODEL_URL, str(model_path))
            if not model_path.exists() or model_path.stat().st_size == 0:
                raise RuntimeError("Downloaded model file is empty.")
        except Exception as exc:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                "Failed to auto-download pose model. "
                "Please check internet or provide --pose-model manually."
            ) from exc

    def load_params_from_json(params_path: str) -> Dict[str, Any]:
        import json

        with open(params_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "coefficients" not in data or "intercept" not in data:
            raise ValueError(
                "Params JSON must contain keys: 'coefficients' and 'intercept'."
            )
        if len(data["coefficients"]) != 4:
            raise ValueError("Params JSON 'coefficients' must contain 4 values.")
        return data

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _landmark_to_xy(self, landmarks: List, idx: int)->Tuple[np.ndarray, float]:
        lm = landmarks[idx]
        point = np.array([lm.x, lm.y], dtype=np.float32)
        visibility = float(getattr(lm, "visibility", 1.0))
        return point, visibility

    def extract_features_from_landmarks(self, landmarks: List)->PostureFeatures:
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
        posture_override_high_risk = label in {"lying_flat", "fetal_like"}
        return {
            "posture_label": label,
            "posture_score": score,
            "is_high_risk": (score >= self.threshold) or posture_override_high_risk,
            "features": {
                "torso_verticality": round(features.torso_verticality, 4),
                "hip_knee_angle": round(features.hip_knee_angle, 2),
                "head_to_floor_distance": round(features.head_to_floor_distance, 4),
                "bounding_box_ratio": round(features.bounding_box_ratio, 4),
            },
            "decision_context": {
                "threshold": self.threshold,
                "score_triggered": score >= self.threshold,
                "posture_override_triggered": posture_override_high_risk,
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
    parser.add_argument("--threshold", type=float, default=0.2, help="Risk threshold")
    parser.add_argument("--params-json",type=str,default=None,help="Path to trained logistic params JSON file.")
    parser.add_argument("--pose-model",type=str,default=None,help="Path to MediaPipe .task model (auto-downloaded if missing).")
    parser.add_argument("--no-auto-download",action="store_true",help="Disable automatic model download when task model is missing.")
    args = parser.parse_args()

    coefficients = None
    intercept = 0.0
    if args.params_json:
        trained_params = PostureRiskDetector.load_params_from_json(args.params_json)
        coefficients = trained_params["coefficients"]
        intercept = trained_params["intercept"]
    detector = PostureRiskDetector(threshold=args.threshold,coefficients=coefficients,intercept=intercept,pose_model_path=args.pose_model,auto_download_model=not args.no_auto_download,)
    try:
        output = detector.analyze_image_path(args.image)
        print(json.dumps(output, indent=2))
    finally:
        detector.close()
