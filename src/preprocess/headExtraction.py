import cv2, os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from src.config import settings


# Anime face detection (Fuyucch1/yolov8_animeface)
script_dir = Path(__file__).parent
model_path = settings.paths.model_dir / 'yolov8x6_animeface.pt'
model = YOLO(str(model_path))

def anime_extraction_recursive(model=model, preserve_dirs=False, device='cpu', cache=False) -> int:
    """
    Extract anime/manga faces from all panels under settings.panels_dir (with subdir support).
    Saves crops under settings.crops_dir.

    Args:
        model: Pre-loaded YOLOv8 model for anime face detection.

    Returns:
        Total number of crops saved.
    """
    panel_root = settings.paths.panels_dir
    crops_root = settings.paths.crops_dir
    Path(crops_root).mkdir(exist_ok=True)

    # Collect all panels recursively
    panel_paths = sorted([
        os.path.join(r, f)
        for r, _, files in os.walk(panel_root)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    count = 0
    for p in tqdm(panel_paths, desc="Detecting faces in panels"):
        # imgsz=512: Input image size for YOLO
        # conf=0.3: Confidence threshold (30% minimum confidence for detection)
        # iou=0.5: IoU threshold for Non-Maximum Suppression (removes overlapping boxes)
        results = model.predict(p, imgsz=512, conf=0.3, iou=0.5, verbose=False, device=device, cache=cache)

        img = cv2.imread(p)

        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            if preserve_dirs == True:
                outp = _make_crop_output_path(p, panel_root, crops_root, f"face_{i}")
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                cv2.imwrite(outp, crop)
            else:
                outp = os.path.join(crops_root, f"crop_{i}.jpg")
                cv2.imwrite(outp, crop)

            count += 1

    print(f"Saved {count} anime face crops to {crops_root}")
    return count

def _make_crop_output_path(panel_path: str, panels_root, crops_root, suffix: str) -> str:
    """Preserve subdirectory structure of panels inside crops_root."""
    rel_path = os.path.relpath(panel_path, panels_root)   # e.g. "ch01/page1_0.jpg"
    rel_root, ext = os.path.splitext(rel_path)
    out_rel = f"{rel_root}_{suffix}{ext}"
    return os.path.join(crops_root, out_rel)
