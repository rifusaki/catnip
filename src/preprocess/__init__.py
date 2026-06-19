"""Preprocessing libraries for catnip dataset unification.

This package contains pure-Python parsers and remappers for the source
annotation formats used by the Stage 1 YOLO training pipeline.

Public submodules:

* :mod:`convert_labels`  — Label Studio per-annotation JSON → YOLO.
* :mod:`manga109_parser` — Manga109 XML → YOLO.
* :mod:`coco_parser`     — COCO JSON (AnimeHeadsv3) → YOLO.
* :mod:`yolo_remap`      — YOLOv8 labels with class remap (0 → 1).
* :mod:`label_studio`    — Label Studio API client (label renaming).
"""
