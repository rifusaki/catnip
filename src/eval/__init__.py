"""Evaluation and benchmarking module for Catnip and peer detection models."""

from src.eval.adapters import (
    BaseDetector,
    CatnipDetector,
    Detection,
    LegacyYoloV5Detector,
    UltralyticsDetector,
    get_detector,
)
from src.eval.metrics import EvaluationMetrics, evaluate_detector, evaluate_single_iou

__all__ = [
    "BaseDetector",
    "CatnipDetector",
    "Detection",
    "EvaluationMetrics",
    "LegacyYoloV5Detector",
    "UltralyticsDetector",
    "evaluate_detector",
    "evaluate_single_iou",
    "get_detector",
]
