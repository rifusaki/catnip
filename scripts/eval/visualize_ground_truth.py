#!/usr/bin/env python3
"""
Visualize Ground Truth Annotations across Catnip Datasets.

Renders high-contrast bounding boxes with class and provenance badges ([GT] vs [NEW]),
illustrating the cross-dataset synthetic enrichment used to train Stage 1 v2.
Generates an interactive HTML gallery and a Markdown summary report in the repo.

Usage:
    pixi run visualize-gt
    pixi run python scripts/eval/visualize_ground_truth.py --output-dir results/gt_gallery --samples-per-dataset 4
"""

import argparse
import html
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color palettes (RGB)
COLOR_BODY = (0, 229, 255)   # Electric Cyan (#00E5FF)
COLOR_HEAD = (255, 145, 0)   # Solar Amber (#FF9100)
COLOR_FACE = (0, 230, 118)   # Neon Emerald (#00E676)

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: COLOR_BODY,
    1: COLOR_HEAD,
    2: COLOR_FACE,
}

CLASS_NAMES: Dict[int, str] = {
    0: "body",
    1: "head",
    2: "face",
}

DATASET_METADATA: Dict[str, dict] = {
    "izutsumi": {
        "title": "Izutsumi (Dungeon Meshi)",
        "domain": "Manga Line-Art",
        "description": "Target character human-annotated ground truth passthrough (body, face).",
        "new_classes": set(),
        "gt_classes": {0, 2},
    },
    "manga109": {
        "title": "Manga109",
        "domain": "Manga Line-Art",
        "description": "Original body (0) and face (2). Synthetic head (1) verified via DeepGHS + Catnip ensemble.",
        "new_classes": {1},
        "gt_classes": {0, 2},
    },
    "ah_yolo": {
        "title": "AnimeHeads YOLO (v1, v2, ahaf)",
        "domain": "Anime Illustration",
        "description": "Original head (1). Synthetic body (0) via person detector + calibrated face (2) via linear adapter.",
        "new_classes": {0, 2},
        "gt_classes": {1},
    },
    "ah_coco": {
        "title": "AnimeHeads COCO (v3)",
        "domain": "Anime Illustration (Danbooru)",
        "description": "Original head (1). Synthetic body (0) via person detector + calibrated face (2) via linear adapter.",
        "new_classes": {0, 2},
        "gt_classes": {1},
    },
}


@dataclass(frozen=True)
class BoundingBox:
    cls_id: int
    xc: float
    yc: float
    w: float
    h: float
    is_new: bool


@dataclass(frozen=True)
class GalleryItem:
    dataset_key: str
    sample_id: str
    image_path: Path
    boxes: List[BoundingBox]
    is_full_page: bool


def load_truetype_font(font_size: int = 13) -> ImageFont.FreeTypeFont:
    """Load a crisp system font for badges, falling back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def parse_label_boxes(label_path: Path, dataset_key: str) -> List[BoundingBox]:
    """Parse YOLO normalized box labels from disk into typed BoundingBox instances."""
    if not label_path.exists():
        return []

    new_classes = DATASET_METADATA.get(dataset_key, {}).get("new_classes", set())
    boxes: List[BoundingBox] = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            is_new = cls_id in new_classes
            boxes.append(BoundingBox(cls_id=cls_id, xc=xc, yc=yc, w=w, h=h, is_new=is_new))

    return boxes


def render_annotated_image(
    image_path: Path,
    boxes: List[BoundingBox],
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """Render high-contrast bounding boxes with class and provenance pills."""
    base_img = Image.open(image_path).convert("RGBA")
    w_img, h_img = base_img.size

    overlay = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Sort boxes by area descending: larger boxes (body) drawn first, smaller boxes (face) on top
    sorted_boxes = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)

    for box in sorted_boxes:
        cls_color = CLASS_COLORS.get(box.cls_id, (255, 255, 255))
        cls_name = CLASS_NAMES.get(box.cls_id, "unknown")

        bw = box.w * w_img
        bh = box.h * h_img
        bx1 = max(0, int((box.xc * w_img) - (bw / 2)))
        by1 = max(0, int((box.yc * h_img) - (bh / 2)))
        bx2 = min(w_img - 1, int(bx1 + bw))
        by2 = min(h_img - 1, int(by1 + bh))

        # Contrast drop shadow / border
        draw.rectangle([bx1 - 1, by1 - 1, bx2 + 1, by2 + 1], outline=(0, 0, 0, 180), width=3)
        draw.rectangle([bx1, by1, bx2, by2], outline=(*cls_color, 255), width=2)

        # Label tag
        tag = f"{cls_name} [NEW]" if box.is_new else f"{cls_name} [GT]"

        bbox = font.getbbox(tag)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x, pad_y = 6, 3
        badge_w = text_w + pad_x * 2
        badge_h = text_h + pad_y * 2

        badge_y1 = by1 - badge_h if by1 - badge_h >= 0 else by1 + 2
        badge_x1 = max(0, min(bx1, w_img - badge_w - 2))
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        # Semi-transparent dark slate badge with border
        badge_bg = (15, 23, 42, 220)
        draw.rounded_rectangle(
            [badge_x1, badge_y1, badge_x2, badge_y2],
            radius=4,
            fill=badge_bg,
            outline=(*cls_color, 230),
            width=1,
        )
        draw.text((badge_x1 + pad_x, badge_y1 + pad_y - 1), tag, font=font, fill=(*cls_color, 255))

    return Image.alpha_composite(base_img, overlay).convert("RGB")


def collect_samples(
    train_lbl_dir: Path,
    train_img_dir: Path,
    staging_dir: Optional[Path],
    samples_per_dataset: int = 4,
    seed: int = 42,
) -> List[GalleryItem]:
    """Select representative slices and full images across all datasets."""
    random.seed(seed)
    items: List[GalleryItem] = []

    dataset_rules = [
        ("manga109", ("manga109_",), 3),
        ("ah_yolo", ("ahv1_", "ahv2_", "ahaf_"), 3),
        ("ah_coco", ("ahv3_",), 3),
        ("izutsumi", ("izutsumi_",), 2),
    ]

    for ds_key, prefixes, min_classes in dataset_rules:
        candidates: List[Tuple[Path, Path, List[BoundingBox]]] = []
        for prefix in prefixes:
            for lbl_p in train_lbl_dir.glob(f"{prefix}*.txt"):
                img_p = train_img_dir / (lbl_p.stem + ".jpg")
                if not img_p.exists():
                    continue
                boxes = parse_label_boxes(lbl_p, ds_key)
                if not boxes:
                    continue
                classes = set(b.cls_id for b in boxes)
                if len(classes) >= min_classes:
                    candidates.append((lbl_p, img_p, boxes))
                    if len(candidates) >= 50:
                        break
            if len(candidates) >= 50:
                break

        random.shuffle(candidates)
        for _, img_p, boxes in candidates[:samples_per_dataset]:
            items.append(
                GalleryItem(
                    dataset_key=ds_key,
                    sample_id=img_p.name,
                    image_path=img_p,
                    boxes=boxes,
                    is_full_page=False,
                )
            )

    # Full page samples if staging directory is present
    if staging_dir and staging_dir.exists():
        for ds_key in ["manga109", "ah_yolo", "ah_coco", "izutsumi"]:
            stage_lbl_dir = staging_dir / ds_key / "labels"
            stage_img_dir = staging_dir / ds_key / "images"
            if not stage_lbl_dir.exists() or not stage_img_dir.exists():
                continue

            full_cands = []
            min_c = 2 if ds_key == "izutsumi" else 3
            for lbl_p in stage_lbl_dir.glob("*.txt"):
                img_p = stage_img_dir / (lbl_p.stem + ".jpg")
                if not img_p.exists():
                    img_p = stage_img_dir / (lbl_p.stem + ".png")
                if not img_p.exists():
                    continue
                boxes = parse_label_boxes(lbl_p, ds_key)
                if not boxes:
                    continue
                classes = set(b.cls_id for b in boxes)
                if len(classes) >= min_c and len(boxes) >= 4:
                    full_cands.append((lbl_p, img_p, boxes))
                    if len(full_cands) >= 20:
                        break

            if full_cands:
                random.shuffle(full_cands)
                _, chosen_img, chosen_boxes = full_cands[0]
                items.append(
                    GalleryItem(
                        dataset_key=ds_key,
                        sample_id=f"FULL_{chosen_img.name}",
                        image_path=chosen_img,
                        boxes=chosen_boxes,
                        is_full_page=True,
                    )
                )

    return items


def write_html_gallery(output_dir: Path, records: List[dict]):
    """Generate interactive, responsive HTML gallery in the repo."""
    cards_html = []
    for r in records:
        tag_type = "Full Canvas" if r["is_full_page"] else "640×640 Slice"
        card = f"""
        <div class="card" data-dataset="{r['dataset']}" data-type="{'full' if r['is_full_page'] else 'slice'}">
          <div class="card-header">
            <span class="badge badge-dataset">{r['title']}</span>
            <span class="badge badge-type">{tag_type}</span>
          </div>
          <div class="img-container">
            <img src="images/{r['filename']}" alt="{html.escape(r['original_name'])}" loading="lazy" />
          </div>
          <div class="card-body">
            <div class="card-title" title="{html.escape(r['original_name'])}">{html.escape(r['original_name'])}</div>
            <div class="card-desc">{html.escape(r['description'])}</div>
            <div class="metrics-grid">
              <div class="metric-pill pill-body">Body: <strong>{r['counts'][0]}</strong></div>
              <div class="metric-pill pill-head">Head: <strong>{r['counts'][1]}</strong></div>
              <div class="metric-pill pill-face">Face: <strong>{r['counts'][2]}</strong></div>
            </div>
            <div class="provenance-bar">
              <span class="tag-provenance tag-new">★ {r['new_count']} Enriched/NEW</span>
              <span class="tag-provenance tag-gt">✓ {r['gt_count']} Original GT</span>
            </div>
          </div>
        </div>
        """
        cards_html.append(card)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Catnip Stage 1 v2 Ground Truth Gallery</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --cyan: #00e5ff;
      --amber: #ff9100;
      --emerald: #00e676;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 32px 24px;
      line-height: 1.5;
    }}
    header {{
      max-width: 1400px;
      margin: 0 auto 32px auto;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
      background: linear-gradient(135deg, #fff 40%, var(--cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    p.subtitle {{
      color: var(--text-muted);
      margin-top: 8px;
      font-size: 15px;
      max-width: 900px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 20px;
      align-items: center;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.04);
      padding: 6px 12px;
      border-radius: 20px;
      border: 1px solid var(--border);
    }}
    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
    .filter-bar {{
      max-width: 1400px;
      margin: 0 auto 28px auto;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .filter-btn {{
      background: #162032;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: var(--cyan);
      color: #050b14;
      border-color: var(--cyan);
      box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }}
    .gallery-grid {{
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 24px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-4px);
      border-color: rgba(0, 229, 255, 0.5);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5);
    }}
    .card-header {{
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
      border-bottom: 1px solid var(--border);
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }}
    .badge-dataset {{
      background: rgba(168, 85, 247, 0.2);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.4);
    }}
    .badge-type {{
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
    }}
    .img-container {{
      width: 100%;
      height: 320px;
      background: #000;
      position: relative;
      overflow: hidden;
      cursor: zoom-in;
    }}
    .img-container img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      transition: transform 0.3s ease;
    }}
    .img-container:hover img {{
      transform: scale(1.05);
    }}
    .card-body {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex-grow: 1;
    }}
    .card-title {{
      font-size: 13px;
      font-weight: 700;
      font-family: ui-monospace, Menlo, Consolas, monospace;
      color: #e2e8f0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .card-desc {{
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.4;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 4px;
    }}
    .metric-pill {{
      font-size: 11px;
      padding: 5px 8px;
      border-radius: 6px;
      text-align: center;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
    }}
    .pill-body strong {{ color: var(--cyan); }}
    .pill-head strong {{ color: var(--amber); }}
    .pill-face strong {{ color: var(--emerald); }}
    .provenance-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      font-weight: 700;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }}
    .tag-new {{ color: #38bdf8; }}
    .tag-gt {{ color: #a3e635; }}
  </style>
</head>
<body>
  <header>
    <h1>Catnip Stage 1 v2 — Ground Truth Enrichment Gallery</h1>
    <p class="subtitle">
      Multi-class Ground Truth annotations used to train the Stage 1 v2 <code>yolo26m</code> detector.
      Demonstrates cross-dataset synthetic enrichment resolving the partial-label contradiction: 
      <code>manga109</code> (synthesized heads), <code>AnimeHeads</code> (synthesized bodies + calibrated faces), and <code>Izutsumi</code> (clean passthrough).
    </p>
    <div class="legend">
      <div class="legend-item"><div class="legend-color" style="background: var(--cyan);"></div> Body (Class 0)</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--amber);"></div> Head (Class 1)</div>
      <div class="legend-item"><div class="legend-color" style="background: var(--emerald);"></div> Face (Class 2)</div>
      <div class="legend-item" style="border-color: #38bdf8; color: #38bdf8;">★ [NEW] Cross-Dataset Synthetic GT</div>
      <div class="legend-item" style="border-color: #a3e635; color: #a3e635;">✓ [GT] Original Ground Truth</div>
    </div>
  </header>

  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterGallery('all')">All Samples ({len(records)})</button>
    <button class="filter-btn" onclick="filterGallery('manga109')">Manga109</button>
    <button class="filter-btn" onclick="filterGallery('ah_yolo')">AnimeHeads YOLO</button>
    <button class="filter-btn" onclick="filterGallery('ah_coco')">AnimeHeads COCO</button>
    <button class="filter-btn" onclick="filterGallery('izutsumi')">Izutsumi</button>
  </div>

  <div class="gallery-grid" id="galleryGrid">
    {''.join(cards_html)}
  </div>

  <script>
    function filterGallery(dataset) {{
      const buttons = document.querySelectorAll('.filter-btn');
      buttons.forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');

      const cards = document.querySelectorAll('.card');
      cards.forEach(card => {{
        if (dataset === 'all' || card.getAttribute('data-dataset') === dataset) {{
          card.style.display = 'flex';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    logger.info("Generated HTML gallery at: %s", output_dir / "index.html")


def write_markdown_report(output_dir: Path, records: List[dict]):
    """Generate repository markdown report with relative image links."""
    md_lines = [
        "# Catnip Stage 1 v2 Ground Truth Gallery",
        "",
        "This directory contains visual samples of the Ground Truth annotations used to train the Stage 1 v2 (`yolo26m`) detector.",
        "",
        "> **Interactive Viewer:** Open [`index.html`](index.html) in your browser for responsive filtering and zoom cards.",
        "",
        "## Class Taxonomy & Color Coding",
        "",
        "- **Class 0 (`body`):** Electric Cyan (`#00E5FF`) — GT on Manga/Izutsumi; `[NEW]` synthesized on AnimeHeads.",
        "- **Class 1 (`head`):** Solar Amber (`#FF9100`) — GT on AnimeHeads; `[NEW]` synthesized on Manga109.",
        "- **Class 2 (`face`):** Neon Emerald (`#00E676`) — GT on Manga/Izutsumi; `[NEW]` calibrated on AnimeHeads.",
        "",
        "## Visual Samples",
        "",
        "| Preview | Dataset | Source ID | Composition | Provenance |",
        "|---|---|---|---|---|",
    ]

    for r in records:
        tag_type = "Full Canvas" if r["is_full_page"] else "640×640 Slice"
        comp_str = f"Body: {r['counts'][0]} · Head: {r['counts'][1]} · Face: {r['counts'][2]}"
        prov_str = f"★ {r['new_count']} NEW · ✓ {r['gt_count']} GT"
        img_md = f"<img src='images/{r['filename']}' width='240' />"
        md_lines.append(
            f"| {img_md} | **{r['title']}** ({tag_type}) | `{r['original_name']}` | {comp_str} | {prov_str} |"
        )

    (output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Generated Markdown summary at: %s", output_dir / "README.md")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual ground truth gallery for Catnip Stage 1 v2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gt_gallery"),
        help="Target output directory (default: results/gt_gallery).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to sliced dataset dir (default: auto-discover stage1_v2_sliced).",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=4,
        help="Number of 640px slices per dataset (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = settings.paths.data

    # Resolve sliced training dataset
    if args.dataset:
        sliced_dir = args.dataset
    else:
        v2_candidate = data_root / "training" / "stage1_v2_sliced"
        if v2_candidate.exists():
            sliced_dir = v2_candidate
        else:
            sliced_dir = settings.paths.stage1_sliced_dir

    train_img_dir = sliced_dir / "images" / "train"
    train_lbl_dir = sliced_dir / "labels" / "train"
    staging_dir = data_root / "data" / "staging_enriched"

    if not train_img_dir.exists() or not train_lbl_dir.exists():
        logger.error(
            "Sliced dataset not found at %s. Please check CATNIP_DATA mount or pass --dataset.",
            sliced_dir,
        )
        sys.exit(1)

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    font = load_truetype_font(13)

    logger.info("Selecting samples from %s (seed=%d)...", sliced_dir, args.seed)
    items = collect_samples(
        train_lbl_dir=train_lbl_dir,
        train_img_dir=train_img_dir,
        staging_dir=staging_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
    )
    logger.info("Selected %d total samples. Rendering boxes...", len(items))

    records: List[dict] = []
    for idx, item in enumerate(items):
        annotated_img = render_annotated_image(item.image_path, item.boxes, font)

        out_name = f"sample_{idx:02d}_{item.dataset_key}_{item.sample_id}"
        if not out_name.endswith(".jpg"):
            out_name += ".jpg"
        out_p = images_dir / out_name
        annotated_img.save(out_p, quality=92)

        counts = {0: 0, 1: 0, 2: 0}
        new_count = 0
        gt_count = 0
        for b in item.boxes:
            counts[b.cls_id] = counts.get(b.cls_id, 0) + 1
            if b.is_new:
                new_count += 1
            else:
                gt_count += 1

        meta = DATASET_METADATA.get(item.dataset_key, {})
        records.append(
            {
                "index": idx,
                "dataset": item.dataset_key,
                "title": meta.get("title", item.dataset_key),
                "domain": meta.get("domain", ""),
                "description": meta.get("description", ""),
                "is_full_page": item.is_full_page,
                "filename": out_name,
                "original_name": item.sample_id,
                "counts": counts,
                "new_count": new_count,
                "gt_count": gt_count,
                "total_boxes": len(item.boxes),
            }
        )

    write_html_gallery(output_dir, records)
    write_markdown_report(output_dir, records)
    logger.info("Successfully published ground truth gallery to %s!", output_dir)


if __name__ == "__main__":
    main()
