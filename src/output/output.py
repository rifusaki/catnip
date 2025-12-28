import numpy as np, json, matplotlib.pyplot as plt, shutil
import cv2
import os
from pathlib import Path
from PIL import Image
from ..recognition.embeddingModel import load_img


def char_nearest_neighbor(crop_paths, final_indices, final_scores, similarity_threshold):
    """
    Display results sorted by similarity score in a grid
    """
    # Dynamic grid sizing based on number of results
    n_results = len(final_indices)
    if n_results <= 8:
        rows, cols = 1, n_results
        figsize = (2 * cols, 2)
    elif n_results <= 16:
        rows, cols = 2, 8
        figsize = (16, 4)
    elif n_results <= 40:
        rows, cols = 5, 8
        figsize = (16, 10)
    else:
        # Show top 120 results for display purposes
        final_indices = final_indices[:120]
        final_scores = final_scores[:120]
        rows, cols = 15, 8
        figsize = (16, 40)
        print(f"Showing top 120 results out of {n_results} matches")

    # Visualization: Display results in grid
    plt.figure(figsize=figsize)
    for i, (idx, score) in enumerate(zip(final_indices, final_scores)):
        # Load and resize crop for display
        im = Image.open(crop_paths[idx]).convert('RGB')
        plt.subplot(rows, cols, i+1)
        plt.imshow(im.resize((128, 128)))
        
        # Show similarity score as title (higher = more similar)
        plt.title(f"{score:.3f}")
        plt.axis('off')

    plt.suptitle(f"Character matches above {similarity_threshold} similarity threshold", fontsize=14)
    plt.tight_layout()
    plt.show()


def save_similar_results(crop_paths, final_indices, OUTPUT_DIR, final_scores=None):
    import os
    import warnings
    if final_scores is not None and len(final_scores) == len(final_indices):
        for idx, score in zip(final_indices, final_scores):
            src = crop_paths[idx]
            # Clean up score for filename
            score_str = f"{score:.4f}"
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            out_name = f"{score_str}_{name}{ext}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            shutil.copy(src, out_path)
    else:
        if final_scores is not None:
            warnings.warn("final_scores provided but length mismatch, saving with original filenames.")
        for index in final_indices:
            shutil.copy(crop_paths[index], OUTPUT_DIR)

def save_inference_results(results_generator, output_dir, inference_source):
    """
    Process YOLO inference results generator, flatten paths, and save images/labels.
    
    Args:
        results_generator: Generator returned by model.predict()
        output_dir: Directory to save results
        inference_source: Source directory used for inference (to calculate relative paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_source = Path(inference_source)
    
    print(f"Saving flattened results to {output_dir}")
    
    for r in results_generator:
        # Calculate flattened filename
        original_path = Path(r.path)
        try:
            rel_path = original_path.relative_to(inference_source)
            # Replace path separators with underscores
            # e.g. v01/001.jpg -> v01_001.jpg
            flat_name = str(rel_path).replace(os.sep, "_")
        except ValueError:
            # Fallback if path is not relative
            flat_name = original_path.name

        # Paths for saving
        save_img_path = output_dir / flat_name
        save_txt_path = save_img_path.with_suffix(".txt")

        # Save Image with Detections
        # r.plot() returns the image as a numpy array (BGR)
        im_array = r.plot() 
        cv2.imwrite(str(save_img_path), im_array)

        # Save Labels if detections exist
        if len(r.boxes) > 0:
            with open(save_txt_path, "w") as f:
                for box in r.boxes:
                    # box.cls is a tensor, get item
                    cls = int(box.cls[0].item())
                    # box.xywhn is a tensor, get list
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    print("Inference results saved.")



