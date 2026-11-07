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
| <img src='images/sample_00_manga109_manga109_TsubasaNoKioku_063_512_512_1152_1152.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_TsubasaNoKioku_063_512_512_1152_1152.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 2 NEW · ✓ 4 GT |
| <img src='images/sample_01_manga109_manga109_KyokugenCyclone_027_0_530_640_1170.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_KyokugenCyclone_027_0_530_640_1170.jpg` | Body: 3 · Head: 1 · Face: 2 | ★ 1 NEW · ✓ 5 GT |
| <img src='images/sample_02_manga109_manga109_Count3DeKimeteAgeru_071_0_530_640_1170.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_Count3DeKimeteAgeru_071_0_530_640_1170.jpg` | Body: 5 · Head: 1 · Face: 3 | ★ 1 NEW · ✓ 8 GT |
| <img src='images/sample_03_manga109_manga109_ToutaMairimasu_031_0_530_640_1170.jpg' width='240' /> | **Manga109** (640×640 Slice) | `manga109_ToutaMairimasu_031_0_530_640_1170.jpg` | Body: 1 · Head: 1 · Face: 1 | ★ 1 NEW · ✓ 2 GT |
| <img src='images/sample_04_ah_yolo_ahv1_7401205_webp.rf.0567d573c639b67d1a7636852cfa315d_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_7401205_webp.rf.0567d573c639b67d1a7636852cfa315d_0_0_640_640.jpg` | Body: 2 · Head: 7 · Face: 4 | ★ 6 NEW · ✓ 7 GT |
| <img src='images/sample_05_ah_yolo_ahv1_1785640_webp.rf.5d27472b3c3e72ab13a73906f6c0a1ea_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_1785640_webp.rf.5d27472b3c3e72ab13a73906f6c0a1ea_0_0_640_640.jpg` | Body: 1 · Head: 3 · Face: 3 | ★ 4 NEW · ✓ 3 GT |
| <img src='images/sample_06_ah_yolo_ahv1_8530089_webp.rf.36fc531f495cad2705f54c0f243aefd4_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_8530089_webp.rf.36fc531f495cad2705f54c0f243aefd4_0_0_640_640.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 4 NEW · ✓ 2 GT |
| <img src='images/sample_07_ah_yolo_ahv1_20240915181409972940_74e18ac01e43dde01a47d2e4ccb1b8c734d4c614_jpg.rf.e84d41ac9e34d0d5619391990cb74319_0_0_640_640.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (640×640 Slice) | `ahv1_20240915181409972940_74e18ac01e43dde01a47d2e4ccb1b8c734d4c614_jpg.rf.e84d41ac9e34d0d5619391990cb74319_0_0_640_640.jpg` | Body: 3 · Head: 4 · Face: 1 | ★ 4 NEW · ✓ 4 GT |
| <img src='images/sample_08_ah_coco_ahv3_00000864_jpg.rf.093aef92e5ddbe1ee9a315edb01d305f_0_0_640_360.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_00000864_jpg.rf.093aef92e5ddbe1ee9a315edb01d305f_0_0_640_360.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 4 NEW · ✓ 2 GT |
| <img src='images/sample_09_ah_coco_ahv3_028_jpg.rf.35ca9fce9c235ad926ef8a271d942ca5_0_0_451_640.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_028_jpg.rf.35ca9fce9c235ad926ef8a271d942ca5_0_0_451_640.jpg` | Body: 8 · Head: 9 · Face: 7 | ★ 15 NEW · ✓ 9 GT |
| <img src='images/sample_10_ah_coco_ahv3_danbooru_6066550_28a22997057d478e9112951e124df699_png.rf.0c0d5483b210d7a938f52ab6220f07d4_0_0_640_395.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_danbooru_6066550_28a22997057d478e9112951e124df699_png.rf.0c0d5483b210d7a938f52ab6220f07d4_0_0_640_395.jpg` | Body: 3 · Head: 6 · Face: 6 | ★ 9 NEW · ✓ 6 GT |
| <img src='images/sample_11_ah_coco_ahv3_00079_png.rf.5d423782406c4fb3f360f949030dc73a_0_0_450_640.jpg' width='240' /> | **AnimeHeads COCO (v3)** (640×640 Slice) | `ahv3_00079_png.rf.5d423782406c4fb3f360f949030dc73a_0_0_450_640.jpg` | Body: 4 · Head: 8 · Face: 4 | ★ 8 NEW · ✓ 8 GT |
| <img src='images/sample_12_izutsumi_izutsumi_v09_0064_518_0_1158_640.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v09_0064_518_0_1158_640.jpg` | Body: 1 · Head: 2 · Face: 2 | ★ 0 NEW · ✓ 5 GT |
| <img src='images/sample_13_izutsumi_izutsumi_v12_019_512_1280_1152_1920.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v12_019_512_1280_1152_1920.jpg` | Body: 1 · Head: 0 · Face: 1 | ★ 0 NEW · ✓ 2 GT |
| <img src='images/sample_14_izutsumi_izutsumi_v11_025_512_1024_1152_1664.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v11_025_512_1024_1152_1664.jpg` | Body: 0 · Head: 1 · Face: 1 | ★ 0 NEW · ✓ 2 GT |
| <img src='images/sample_15_izutsumi_izutsumi_v12_110_512_0_1152_640.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (640×640 Slice) | `izutsumi_v12_110_512_0_1152_640.jpg` | Body: 0 · Head: 1 · Face: 1 | ★ 0 NEW · ✓ 2 GT |
| <img src='images/sample_16_manga109_FULL_manga109_UnbalanceTokyo_036.jpg' width='240' /> | **Manga109** (Full Canvas) | `FULL_manga109_UnbalanceTokyo_036.jpg` | Body: 10 · Head: 5 · Face: 9 | ★ 5 NEW · ✓ 19 GT |
| <img src='images/sample_17_ah_yolo_FULL_ahv1_7513289_webp.rf.4efd98f89c818c6d1d686dc849106381.jpg' width='240' /> | **AnimeHeads YOLO (v1, v2, ahaf)** (Full Canvas) | `FULL_ahv1_7513289_webp.rf.4efd98f89c818c6d1d686dc849106381.jpg` | Body: 3 · Head: 4 · Face: 4 | ★ 7 NEW · ✓ 4 GT |
| <img src='images/sample_18_ah_coco_FULL_ahv3_DnCj0QDVAAALtHh_jpg.rf.c8a99b8af35e455f03bf5af819baca9c.jpg' width='240' /> | **AnimeHeads COCO (v3)** (Full Canvas) | `FULL_ahv3_DnCj0QDVAAALtHh_jpg.rf.c8a99b8af35e455f03bf5af819baca9c.jpg` | Body: 2 · Head: 2 · Face: 2 | ★ 4 NEW · ✓ 2 GT |
| <img src='images/sample_19_izutsumi_FULL_izutsumi_v06_00008.jpg' width='240' /> | **Izutsumi (Dungeon Meshi)** (Full Canvas) | `FULL_izutsumi_v06_00008.jpg` | Body: 0 · Head: 4 · Face: 4 | ★ 0 NEW · ✓ 8 GT |