import numpy as np, json, matplotlib.pyplot as plt, shutil
from PIL import Image
from .embeddingModel import load_img


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


