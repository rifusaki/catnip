# Catnip Stage 1 v2 Ground Truth Gallery

This directory contains visual samples of the Ground Truth annotations used to train the Stage 1 v2 (`yolo26m`) detector.

> **Interactive Viewer:** Open [`index.html`](index.html) in your browser for responsive filtering and zoom cards.

## Class Taxonomy & Color Coding

- **Class 0 (`body`):** Electric Cyan (`#00E5FF`) — GT on Manga/Izutsumi; `[NEW]` synthesized on AnimeHeads.
- **Class 1 (`head`):** Solar Amber (`#FF9100`) — GT on AnimeHeads; `[NEW]` synthesized on Manga109.
- **Class 2 (`face`):** Neon Emerald (`#00E676`) — GT on Manga/Izutsumi; `[NEW]` calibrated on AnimeHeads.

## Visual Samples

| Preview | Dataset | Source ID | Composition | Provenance |
|---|---|---|---|---|
| <img src='images/sample_00_manga109_manga109_UchuKigekiM774_039_512_0_1152_640.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_UchuKigekiM774_039_512_0_1152_640.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 2 NEW · ✓ 4 GT |
| <img src='images/sample_01_manga109_manga109_OhWareraRettouSeitokai_083_512_0_1152_640.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_OhWareraRettouSeitokai_083_512_0_1152_640.jpg` | Body: 4 · Head: 4 · Face: 4 | ★ 4 NEW · ✓ 8 GT |
| <img src='images/sample_02_manga109_manga109_Count3DeKimeteAgeru_071_0_530_640_1170.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_Count3DeKimeteAgeru_071_0_530_640_1170.jpg` | Body: 5 · Head: 1 · Face: 3 | ★ 1 NEW · ✓ 8 GT |
| <img src='images/sample_03_manga109_manga109_ToutaMairimasu_031_0_530_640_1170.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_ToutaMairimasu_031_0_530_640_1170.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 1 NEW · ✓ 2 GT |
| <img src='images/sample_04_ah_yolo_ahv1_20240915182518701720_a82240eb8823655b050874a9de84c6716a730441_jpg.rf.125bb0dd6209f47a134581f157e8c835_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_20240915182518701720_a82240eb8823655b050874a9de84c6716a730441_jpg.rf.125bb0dd6209f47a134581f157e8c835_0_0_640_640.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_05_ah_yolo_ahv1_2074439_webp.rf.e162e31cf75e90272231565af9fb47da_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_2074439_webp.rf.e162e31cf75e90272231565af9fb47da_0_0_640_640.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_06_ah_yolo_ahv1_20240915174556946760_d3ccf4b3ef4576b4265dccbe902c725a88773c3e_jpg.rf.9c02b907f261b52da6bf95d4155d2207_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_20240915174556946760_d3ccf4b3ef4576b4265dccbe902c725a88773c3e_jpg.rf.9c02b907f261b52da6bf95d4155d2207_0_0_640_640.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_07_ah_yolo_ahv1_20240915174328552147_4b3b07e995af008b8bad8676bc2855f0f49aaf4f_jpg.rf.a30e206afc38b0d35579fff7ba68ec54_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_20240915174328552147_4b3b07e995af008b8bad8676bc2855f0f49aaf4f_jpg.rf.a30e206afc38b0d35579fff7ba68ec54_0_0_640_640.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_08_ah_coco_ahv3_Photo-Apr-13-7-06-15-PM_png.rf.2d23d8ab9fbc95e9c9f3ab4c87c48df4_0_0_640_359.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_Photo-Apr-13-7-06-15-PM_png.rf.2d23d8ab9fbc95e9c9f3ab4c87c48df4_0_0_640_359.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_09_ah_coco_ahv3_00020592_jpg.rf.799fe9e7006b29b2a65f51ed97b46caa_0_0_640_360.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_00020592_jpg.rf.799fe9e7006b29b2a65f51ed97b46caa_0_0_640_360.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 2 NEW · ✓ 1 GT |
| <img src='images/sample_10_ah_coco_ahv3_danbooru_6068770_24550b44e092c3b32cce79b50f7aa67e_jpg.rf.439539203dbed9c7656e8d99c80fe7fb_0_0_452_640.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_danbooru_6068770_24550b44e092c3b32cce79b50f7aa67e_jpg.rf.439539203dbed9c7656e8d99c80fe7fb_0_0_452_640.jpg` | Body: 5 · Head: 6 · Face: 6 | ★ 11 NEW · ✓ 6 GT |
| <img src='images/sample_11_ah_coco_ahv3_Higurashi-When-They-Cry-Dice-Killing-Arc-c001-v00-p013-dig-Yen-Press-LuCaZ-_jpg.rf.a05ec35fa9531d7b8f9a83b48d9349c0_0_0_426_640.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_Higurashi-When-They-Cry-Dice-Killing-Arc-c001-v00-p013-dig-Yen-Press-LuCaZ-_jpg.rf.a05ec35fa9531d7b8f9a83b48d9349c0_0_0_426_640.jpg` | Body: 3 · Head: 4 · Face: 2 | ★ 5 NEW · ✓ 4 GT |
| <img src='images/sample_12_izutsumi_izutsumi_v09_0070_518_512_1158_1152.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v09_0070_518_512_1158_1152.jpg` | Body: 1 · Head: 0 · Face: 1 | ★ 0 NEW · ✓ 2 GT |
| <img src='images/sample_13_izutsumi_izutsumi_v09_0072_0_978_640_1618.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v09_0072_0_978_640_1618.jpg` | Body: 2 · Head: 0 · Face: 2 | ★ 0 NEW · ✓ 4 GT |
| <img src='images/sample_14_izutsumi_izutsumi_v09_0054_512_0_1152_640.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v09_0054_512_0_1152_640.jpg` | Body: 1 · Head: 0 · Face: 2 | ★ 0 NEW · ✓ 3 GT |
| <img src='images/sample_15_izutsumi_izutsumi_v09_0054_0_0_640_640.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v09_0054_0_0_640_640.jpg` | Body: 1 · Head: 0 · Face: 1 | ★ 0 NEW · ✓ 2 GT |
| <img src='images/sample_16_manga109_FULL_manga109_UnbalanceTokyo_036.jpg' width='240' /> | **Manga109** (Full Canvas) | `FULL_manga109_UnbalanceTokyo_036.jpg` | Body: 10 · Head: 5 · Face: 9 | ★ 5 NEW · ✓ 19 GT |
| <img src='images/sample_17_ah_yolo_FULL_ahv1_7513289_webp.rf.4efd98f89c818c6d1d686dc849106381.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (Full Canvas) | `FULL_ahv1_7513289_webp.rf.4efd98f89c818c6d1d686dc849106381.jpg` | Body: 3 · Head: 4 · Face: 4 | ★ 7 NEW · ✓ 4 GT |
| <img src='images/sample_18_ah_coco_FULL_ahv3_DnCj0QDVAAALtHh_jpg.rf.c8a99b8af35e455f03bf5af819baca9c.jpg' width='240' /> | **AnimeHeads COCO (v3)** (Full Canvas) | `FULL_ahv3_DnCj0QDVAAALtHh_jpg.rf.c8a99b8af35e455f03bf5af819baca9c.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 4 NEW · ✓ 2 GT |
| <img src='images/sample_19_izutsumi_FULL_izutsumi_v09_0066.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (Full Canvas) | `FULL_izutsumi_v09_0066.jpg` | Body: 2 · Head: 0 · Face: 2 | ★ 0 NEW · ✓ 4 GT |