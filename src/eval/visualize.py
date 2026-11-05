"""Visualization module for rendering side-by-side model comparison grids."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from src.eval.adapters import Detection

logger = logging.getLogger(__name__)

COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "Ground Truth": (0, 220, 0),        # Vibrant Green
    "Catnip-Stage1": (255, 180, 0),     # Bright Cyan/Sky (BGR)
    "fuyucchi_face": (220, 0, 220),     # Magenta
    "zymk9_face": (0, 140, 255),        # Deep Orange
    "deepghs_face_nano": (0, 230, 255), # Yellow
    "deepghs_head_nano": (255, 100, 100), # Light Violet/Blue
}


def _draw_boxes_on_image(
    base_image: np.ndarray,
    boxes: Sequence[Detection],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw bounding boxes and a title banner onto a copy of an image."""
    img = base_image.copy()
    h, w = img.shape[:2]

    # Draw title banner
    banner_height = max(32, int(h * 0.06))
    cv2.rectangle(img, (0, 0), (w, banner_height), (30, 30, 30), -1)
    font_scale = banner_height / 45.0
    cv2.putText(
        img,
        title,
        (10, int(banner_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, int(font_scale * 2)),
        cv2.LINE_AA,
    )

    thickness = max(2, int(min(h, w) / 250))

    for box in boxes:
        x1, y1, x2, y2 = box.to_pixel_coords(w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label with confidence score
        label = f"{box.score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, 1)
        tag_y = max(banner_height + lh + 4, y1)
        cv2.rectangle(img, (x1, tag_y - lh - 4), (x1 + lw + 4, tag_y + 2), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def render_comparison_grid(
    image: np.ndarray,
    ground_truth_boxes: Sequence[Detection],
    predictions_by_model: Mapping[str, Sequence[Detection]],
    output_path: Path,
) -> Path:
    """Compose a horizontal comparison panel comparing Ground Truth against each model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []

    # 1. Ground Truth Panel
    gt_panel = _draw_boxes_on_image(
        base_image=image,
        boxes=ground_truth_boxes,
        title=f"Ground Truth ({len(ground_truth_boxes)})",
        color=COLOR_PALETTE["Ground Truth"],
    )
    panels.append(gt_panel)

    # 2. Model Prediction Panels
    for model_name, preds in predictions_by_model.items():
        color = COLOR_PALETTE.get(model_name, (200, 200, 200))
        panel = _draw_boxes_on_image(
            base_image=image,
            boxes=preds,
            title=f"{model_name} ({len(preds)})",
            color=color,
        )
        panels.append(panel)

    # Stitch panels horizontally
    composite = np.hstack(panels)
    cv2.imwrite(str(output_path), composite)
    logger.info("Saved visual comparison to %s", output_path)
    return output_path
