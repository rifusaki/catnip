import sys
from pathlib import Path
from PIL import Image

import pytest

from scripts.eval.visualize_ground_truth import (
    BoundingBox,
    parse_label_boxes,
    render_annotated_image,
    load_truetype_font,
    write_html_gallery,
    write_markdown_report,
)


def test_parse_label_boxes(tmp_path: Path):
    lbl_file = tmp_path / "sample.txt"
    lbl_file.write_text(
        "0 0.5 0.5 0.2 0.4\n"
        "1 0.3 0.3 0.1 0.1\n"
        "2 0.3 0.3 0.05 0.05\n"
    )
    # Test for manga109 (head is new, body and face are GT)
    boxes = parse_label_boxes(lbl_file, "manga109")
    assert len(boxes) == 3
    assert boxes[0].cls_id == 0 and not boxes[0].is_new
    assert boxes[1].cls_id == 1 and boxes[1].is_new
    assert boxes[2].cls_id == 2 and not boxes[2].is_new

    # Test for ah_yolo (body and face are new, head is GT)
    boxes_ah = parse_label_boxes(lbl_file, "ah_yolo")
    assert boxes_ah[0].cls_id == 0 and boxes_ah[0].is_new
    assert boxes_ah[1].cls_id == 1 and not boxes_ah[1].is_new
    assert boxes_ah[2].cls_id == 2 and boxes_ah[2].is_new


def test_render_annotated_image(tmp_path: Path):
    img_path = tmp_path / "test.jpg"
    base = Image.new("RGB", (640, 640), color=(200, 200, 200))
    base.save(img_path)

    boxes = [
        BoundingBox(cls_id=0, xc=0.5, yc=0.5, w=0.4, h=0.6, is_new=False),
        BoundingBox(cls_id=1, xc=0.5, yc=0.3, w=0.2, h=0.2, is_new=True),
        BoundingBox(cls_id=2, xc=0.5, yc=0.32, w=0.1, h=0.1, is_new=False),
    ]
    font = load_truetype_font(12)
    annotated = render_annotated_image(img_path, boxes, font)

    assert annotated.size == (640, 640)
    assert annotated.mode == "RGB"


def test_write_html_and_markdown(tmp_path: Path):
    out_dir = tmp_path / "gallery"
    out_dir.mkdir()
    records = [
        {
            "index": 0,
            "dataset": "manga109",
            "title": "Manga109",
            "domain": "Manga Line-Art",
            "description": "Test description",
            "is_full_page": False,
            "filename": "sample_0.jpg",
            "original_name": "sample_0.jpg",
            "counts": {0: 1, 1: 1, 2: 1},
            "new_count": 1,
            "gt_count": 2,
            "total_boxes": 3,
        }
    ]
    write_html_gallery(out_dir, records)
    write_markdown_report(out_dir, records)

    assert (out_dir / "index.html").exists()
    assert (out_dir / "README.md").exists()
    assert "Manga109" in (out_dir / "index.html").read_text()
    assert "sample_0.jpg" in (out_dir / "README.md").read_text()
