# Adenzu Panel Extraction – Developer API Reference

This module provides tools for **manga and comic panel segmentation**, offering both **traditional computer vision methods** and **AI-based detection**. Images are represented as OpenCV `numpy.ndarray`s unless otherwise noted.

---

## High-Level API

### `extract_panels_for_image(image_path: str, output_dir: str, fallback: bool = True, split_joint_panels: bool = False, mode: str = OutputMode.BOUNDING, merge: str = MergeMode.NONE) -> None`
Extracts panels from a single image file and saves them into the specified directory.

- **Parameters**  
  - `image_path`: Path to the input image.  
  - `output_dir`: Directory to save panel images.  
  - `fallback`: Use threshold-based extraction if contour-based fails.  
  - `split_joint_panels`: Attempt to split “connected” panels.  
  - `mode`: `OutputMode.BOUNDING` or `OutputMode.MASKED`.  
  - `merge`: `MergeMode.NONE`, `MergeMode.HORIZONTAL`, or `MergeMode.VERTICAL`.  

---

### `extract_panels_for_images_in_folder(input_dir: str, output_dir: str, fallback: bool = True, split_joint_panels: bool = False, mode: str = OutputMode.BOUNDING, merge: str = MergeMode.NONE) -> tuple[int, int]`
Processes an entire folder of images, saving panels for each.

- **Returns**: `(num_files_processed, num_panels_extracted)`  

---

### `extract_panels_for_images_in_folder_by_ai(input_dir: str, output_dir: str) -> tuple[int, int]`
Like above, but uses the trained neural network model rather than CV heuristics.

---

## Mid-Level Programmatic API

Use these functions when working with in-memory images.

### `generate_panel_blocks(image: np.ndarray, background_generator: Callable[[np.ndarray], np.ndarray] = generate_background_mask, split_joint_panels: bool = False, fallback: bool = True, mode: str = OutputMode.BOUNDING, merge: str = MergeMode.NONE, rtl_order: bool = False) -> list[np.ndarray]`
Segment a page into panels using contour-based extraction.

- **Parameters**  
  - `image`: Input image (BGR array).  
  - `background_generator`: Function that produces a background mask. Defaults to `generate_background_mask`.  
  - `split_joint_panels`: Try to split panels that are stuck together.  
  - `fallback`: Use threshold-based extraction if needed.  
  - `mode`: See `OutputMode`.  
  - `merge`: See `MergeMode`.  
  - `rtl_order`: If `True`, panels are sorted right-to-left (manga style).  

- **Returns**: List of panel images as numpy arrays.

---

### `generate_panel_blocks_by_ai(image: np.ndarray, merge: str = MergeMode.NONE, rtl_order: bool = False) -> list[np.ndarray]`
Segment a page into panels using the built-in AI model.

- **Returns**: List of panel images as numpy arrays.

---

## Supporting Enums

### `OutputMode`
- `OutputMode.BOUNDING`: Rectangular crops of panels.  
- `OutputMode.MASKED`: Masked crops with background filled in.  
- `OutputMode.from_index(index: int) -> str`: Map integer to mode.  

### `MergeMode`
- `MergeMode.NONE`: Return each panel separately.  
- `MergeMode.VERTICAL`: Merge vertically aligned panels.  
- `MergeMode.HORIZONTAL`: Merge horizontally aligned panels.  
- `MergeMode.from_index(index: int) -> str`: Map integer to mode.  

---

## Lower-Level Utilities (optional use)

- `extract_panels(image, contours, ...)`  
  Cut out panels from contours.  
- `generate_background_mask(grayscale_image)`  
  Build a binary mask for the page background.  
- `preprocess_image(grayscale_image)` / `preprocess_image_with_dilation(grayscale_image)`  
  Image preprocessing for contour extraction.  
- `threshold_extraction(image, grayscale_image, mode=...)`  
  Threshold-based fallback segmentation.  
- `get_page_without_background(grayscale_image, background_mask, split_joint_panels=False)`  
  Strip background pixels from an image.  

These are useful if you want to experiment with new segmentation pipelines.

---

## Example Usage

```python
import cv2
from adenzu_panel import panel

# Load image with OpenCV
image = cv2.imread("page.png")

# Extract panels in memory
panels = panel.generate_panel_blocks(image, rtl_order=True)

for i, p in enumerate(panels):
    cv2.imwrite(f"panel_{i}.png", p)

# Or use AI-based detection
ai_panels = panel.generate_panel_blocks_by_ai(image)
```
