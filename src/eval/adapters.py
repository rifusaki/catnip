"""Model adapters for unifying diverse YOLO detection backends into a standard interface."""

from __future__ import annotations

import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "catnip" / "benchmarks"

BENCHMARK_CATALOG: dict[str, dict[str, str]] = {
    "fuyucchi_face": {
        "url": "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt",
        "filename": "yolov8x6_animeface.pt",
        "arch": "YOLOv8x6 (1280px)",
        "target": "face",
        "type": "ultralytics",
        "default_imgsz": "1280",
    },
    "zymk9_face": {
        "url": "https://github.com/zymk9/yolov5_anime/raw/master/weights/yolov5s_anime.pt",
        "filename": "yolov5s_anime.pt",
        "arch": "YOLOv5s (640px)",
        "target": "face",
        "type": "yolov5",
        "default_imgsz": "640",
    },
    "deepghs_face_nano": {
        "url": "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/model.pt",
        "filename": "deepghs_face_v1.4_n.pt",
        "arch": "YOLOv8n (640px)",
        "target": "face",
        "type": "ultralytics",
        "default_imgsz": "640",
    },
    "deepghs_head_nano": {
        "url": "https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_n_yv11/model.pt",
        "filename": "deepghs_head_v2.0_n_yv11.pt",
        "arch": "YOLO11n (640px)",
        "target": "head",
        "type": "ultralytics",
        "default_imgsz": "640",
    },
    "deepghs_head_x": {
        "url": "https://huggingface.co/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_x_yv11/model.pt",
        "filename": "deepghs_head_v2.0_x_yv11.pt",
        "arch": "YOLO11x (640px)",
        "target": "head",
        "type": "ultralytics",
        "default_imgsz": "640",
    },
    # NOTE: deepghs publishes no x-sized face/person checkpoints — the
    # largest existing variants are face_detect_v1.4_s and
    # person_detect_v1.2_s (verified via the HF repo trees on 2026-11-06;
    # the plan's *_x URLs return 404).
    "deepghs_face_s": {
        "url": "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_s/model.pt",
        "filename": "deepghs_face_v1.4_s.pt",
        "arch": "YOLOv8s (640px)",
        "target": "face",
        "type": "ultralytics",
        "default_imgsz": "640",
    },
    "deepghs_person_s": {
        "url": "https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.2_s/model.pt",
        "filename": "deepghs_person_v1.2_s.pt",
        "arch": "YOLOv8s (640px)",
        "target": "body",
        "type": "ultralytics",
        "default_imgsz": "640",
    },
}


# ============================================================================
# Core Data Models (Law 2 — Make Illegal States Unrepresentable)
# ============================================================================

@dataclass(frozen=True, slots=True)
class Detection:
    """Normalized bounding box detection with confidence and class."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    def __post_init__(self) -> None:
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError(f"Invalid bounding box coordinates: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Confidence score {self.score} must be within [0.0, 1.0]")

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized [0, 1] coordinates to pixel integers."""
        px1 = int(round(np.clip(self.x1, 0.0, 1.0) * width))
        py1 = int(round(np.clip(self.y1, 0.0, 1.0) * height))
        px2 = int(round(np.clip(self.x2, 0.0, 1.0) * width))
        py2 = int(round(np.clip(self.y2, 0.0, 1.0) * height))
        return px1, py1, max(px1, px2), max(py1, py2)


# ============================================================================
# Base Model Adapter
# ============================================================================

class BaseDetector(ABC):
    """Abstract interface for all evaluated detection backends."""

    def __init__(self, name: str, architecture: str, device: str = "cpu") -> None:
        self.name = name
        self.architecture = architecture
        self.device = self._resolve_device(device)
        self.parameter_count: int = 0
        self.weight_size_bytes: int = 0

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = 0.3) -> list[Detection]:
        """Run inference on an RGB/BGR numpy image and return normalized detections."""
        pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        dev = device.lower().strip()
        if dev == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS device requested but not available. Falling back to CPU.")
            return "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            return "cpu"
        return dev


# ============================================================================
# Ultralytics Adapter (YOLO26, YOLOv8, YOLO11)
# ============================================================================

class UltralyticsDetector(BaseDetector):
    """Adapter for models running on Ultralytics (YOLOv8, YOLO11, and peers)."""

    def __init__(
        self,
        name: str,
        model_path: Path | str,
        architecture: str,
        device: str = "cpu",
        imgsz: int = 640,
        target_class_id: int | None = 0,
    ) -> None:
        super().__init__(name=name, architecture=architecture, device=device)
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.weight_size_bytes = self.model_path.stat().st_size
        self.imgsz = imgsz
        self.target_class_id = target_class_id

        logger.info("Loading %s on %s from %s", self.name, self.device, self.model_path)
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        try:
            self.parameter_count = sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            self.parameter_count = 0

    def predict(self, image: np.ndarray, conf: float = 0.3) -> list[Detection]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(xyxy, scores, classes):
            if self.target_class_id is not None and cls_id != self.target_class_id:
                continue

            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    x1=float(np.clip(x1 / w, 0.0, 1.0)),
                    y1=float(np.clip(y1 / h, 0.0, 1.0)),
                    x2=float(np.clip(x2 / w, 0.0, 1.0)),
                    y2=float(np.clip(y2 / h, 0.0, 1.0)),
                    score=float(score),
                    class_id=0,
                )
            )

        return detections


# ============================================================================
# Catnip Multi-Part Adapter (Remaps Stage 1 classes to target benchmark)
# ============================================================================

class CatnipDetector(UltralyticsDetector):
    """Catnip Stage 1 detector with automatic class isolation and remapping."""

    CATNIP_CLASSES: ClassVar[dict[str, int]] = {
        "body": 0,
        "head": 1,
        "face": 2,
    }

    def __init__(
        self,
        model_path: Path | str,
        target: str = "face",
        device: str = "cpu",
        imgsz: int = 640,
    ) -> None:
        norm_target = target.lower().strip()
        if norm_target not in self.CATNIP_CLASSES:
            raise ValueError(
                f"Invalid Catnip evaluation target '{target}'. Expected one of {list(self.CATNIP_CLASSES.keys())}"
            )

        catnip_class_id = self.CATNIP_CLASSES[norm_target]
        super().__init__(
            name="Catnip-Stage1",
            model_path=model_path,
            architecture="YOLO26 Nano (640px SAHI-parity)",
            device=device,
            imgsz=imgsz,
            target_class_id=catnip_class_id,
        )
        self.target = norm_target
        logger.info(
            "Catnip detector initialized for target '%s' (mapping source class %d -> evaluated class 0)",
            self.target,
            catnip_class_id,
        )


# ============================================================================
# Legacy YOLOv5 Adapter (zymk9)
# ============================================================================

class LegacyYoloV5Detector(BaseDetector):
    """Adapter for legacy YOLOv5 models using torch.hub."""

    def __init__(
        self,
        name: str,
        model_path: Path | str,
        architecture: str = "YOLOv5s (640px)",
        device: str = "cpu",
    ) -> None:
        super().__init__(name=name, architecture=architecture, device=device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Legacy YOLOv5 model not found: {self.model_path}")

        self.weight_size_bytes = self.model_path.stat().st_size
        logger.info("Loading legacy YOLOv5 %s via torch.hub on %s", self.name, self.device)

        # YOLOv5's select_device expects '0' or 'cpu' for devices; passing literal
        # 'cuda' triggers a bug in YOLOv5 that sets os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda',
        # poisoning the process and breaking CUDA for all subsequent models.
        hub_device = "0" if self.device.startswith("cuda") else ("cpu" if self.device == "mps" else self.device)
        orig_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(self.model_path),
                trust_repo=True,
                device=hub_device,
            )
            if self.device != hub_device:
                try:
                    self.model.to(self.device)
                except Exception as to_err:
                    logger.warning("Could not move legacy YOLOv5 to %s: %s", self.device, to_err)
        except Exception as err:
            logger.error("Failed to load legacy YOLOv5 model via torch.hub: %s", err)
            raise
        finally:
            # Law 3 — Atomic Predictability: Restore or sanitize CUDA_VISIBLE_DEVICES
            if orig_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_visible
            elif os.environ.get("CUDA_VISIBLE_DEVICES") == "cuda":
                del os.environ["CUDA_VISIBLE_DEVICES"]

        try:
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
        except Exception:
            self.parameter_count = 0

    def predict(self, image: np.ndarray, conf: float = 0.3) -> list[Detection]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        self.model.conf = conf
        # YOLOv5 expects RGB image
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = image

        results = self.model(rgb_img, size=640)
        detections: list[Detection] = []

        pred_tensor = results.xyxy[0].cpu().numpy()
        for row in pred_tensor:
            x1, y1, x2, y2, score, _ = row
            detections.append(
                Detection(
                    x1=float(np.clip(x1 / w, 0.0, 1.0)),
                    y1=float(np.clip(y1 / h, 0.0, 1.0)),
                    x2=float(np.clip(x2 / w, 0.0, 1.0)),
                    y2=float(np.clip(y2 / h, 0.0, 1.0)),
                    score=float(score),
                    class_id=0,
                )
            )

        return detections


# ============================================================================
# Model Downloader & Factory Functions
# ============================================================================

def download_benchmark_model(model_key: str, cache_dir: Path | None = None, timeout: float = 60.0) -> Path:
    """Download a benchmark model from the catalog and cache it locally with atomic write."""
    if model_key not in BENCHMARK_CATALOG:
        raise KeyError(f"Unknown benchmark model '{model_key}'. Available: {list(BENCHMARK_CATALOG.keys())}")

    target_dir = cache_dir or DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    info = BENCHMARK_CATALOG[model_key]
    dest_path = target_dir / info["filename"]
    tmp_path = dest_path.with_suffix(".tmp")

    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Using cached model: %s", dest_path)
        return dest_path

    logger.info("Downloading %s from %s ...", model_key, info["url"])
    req = urllib.request.Request(info["url"], headers={"User-Agent": "Catnip-Benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
        tmp_path.replace(dest_path)
        logger.info("Downloaded %s (%.2f MB)", model_key, dest_path.stat().st_size / 1e6)
    except Exception as err:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download benchmark model {model_key}: {err}") from err

    return dest_path


def get_detector(
    name: str,
    target: str = "face",
    custom_path: Path | str | None = None,
    device: str = "cpu",
) -> BaseDetector:
    """Factory creating an initialized detector instance by model name or key."""
    norm_name = name.lower().strip()

    # Case 1: Catnip Stage 1 model
    if norm_name in ("catnip", "catnip-stage1", "catnip_stage1", "catnip_nano"):
        if custom_path is None:
            from src.config import settings

            candidates = [
                # 1. Full training deliverable (Colab / production run)
                settings.paths.model_dir / "yolo26_stage1_body_head_face.pt",
                # 2. Raw best weights from full training run
                settings.paths.runs_dir / "detect" / "stage1" / "weights" / "best.pt",
                # 3. Last rolling checkpoint from full training run
                settings.paths.runs_dir / "detect" / "stage1" / "weights" / "last.pt",
                # 4. Local repository models directory (full model)
                REPO_ROOT / "models" / "yolo26_stage1_body_head_face.pt",
                # 5. Local repository models directory (M3 subset baseline deliverable)
                REPO_ROOT / "models" / "yolo26_stage1_body_head_face_m3.pt",
                # 6. M3 subset run weights
                settings.paths.runs_dir / "detect" / "stage1_m3" / "weights" / "best.pt",
                REPO_ROOT / "runs" / "detect" / "stage1_m3" / "weights" / "best.pt",
                # 7. Cached benchmark model
                DEFAULT_CACHE_DIR / "catnip_stage1.pt",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if not found:
                search_list = "\n  - ".join(str(p) for p in candidates)
                raise FileNotFoundError(
                    f"Catnip checkpoint not provided via --catnip-model and no checkpoint found in candidates:\n  - {search_list}"
                )
            custom_path = found

        return CatnipDetector(model_path=custom_path, target=target, device=device)

    # Case 2: Standard Benchmark Catalog Models
    if norm_name in BENCHMARK_CATALOG:
        info = BENCHMARK_CATALOG[norm_name]
        if info["target"] != target:
            raise ValueError(
                f"Model '{norm_name}' is trained for target '{info['target']}', "
                f"cannot be used for benchmark target '{target}'."
            )

        model_path = Path(custom_path) if custom_path else download_benchmark_model(norm_name)

        if info["type"] == "ultralytics":
            return UltralyticsDetector(
                name=norm_name,
                model_path=model_path,
                architecture=info["arch"],
                device=device,
                imgsz=int(info.get("default_imgsz", 640)),
                target_class_id=0,
            )
        elif info["type"] == "yolov5":
            return LegacyYoloV5Detector(
                name=norm_name,
                model_path=model_path,
                architecture=info["arch"],
                device=device,
            )

    raise ValueError(
        f"Unknown model name '{name}'. Must be 'catnip' or one of {list(BENCHMARK_CATALOG.keys())}"
    )
