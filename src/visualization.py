import numpy as np, json, matplotlib.pyplot as plt, shutil
from PIL import Image
from .embeddingModel import load_img

def izutsumi_query(EMBED_PATH, CROP_PATH, IMG_SIZE, embed_model, seed_paths, similarity_threshold=0.7):
    """
    Find similar character crops using similarity threshold in embedding space.
    
    This function implements threshold-based similarity search:
    1. Load precomputed embeddings database
    2. Process seed images to create query embedding
    3. Find all crops above similarity threshold using cosine distance
    
    Args:
        EMBED_PATH: Path to saved embeddings (.npy file)
        CROP_PATH: Path to saved crop paths (.json file) 
        IMG_SIZE: Image size for preprocessing seed images
        embed_model: Trained embedding model
        seed_paths: List of paths to seed/example character images
        similarity_threshold: Minimum cosine similarity (0-1, higher = more similar)

    Returns: 
        crop_paths
        final_indices: iterate over crop_paths[final_indices] to locate crop
        final_scores
    """
    
    # Load precomputed embeddings database
    embs = np.load(EMBED_PATH)
    with open(CROP_PATH, 'r') as f:
        crop_paths = json.load(f)

    if len(seed_paths) > 0:
        # Process seed images to create query embedding
        # Multiple seeds allow for better character representation
        X_seed = np.stack([load_img(p, IMG_SIZE) for p in seed_paths], axis=0)
        seed_embs = embed_model.predict(X_seed, batch_size=8)
        
        # Average multiple seed embeddings for robust query representation
        # This handles variations in pose, lighting, etc. across seed images
        query_vec = np.mean(seed_embs, axis=0, keepdims=True)  # average embedding
        
        # Re-normalize after averaging (maintains unit length property)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Calculate cosine similarities with all embeddings
        # Higher values = more similar (cosine similarity ranges from -1 to 1)
        similarities = np.dot(embs, query_vec.T).flatten()
        
        # Filter by similarity threshold
        similar_indices = np.where(similarities >= similarity_threshold)[0]
        similar_scores = similarities[similar_indices]
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similar_scores)[::-1]
        final_indices = similar_indices[sorted_indices]
        final_scores = similar_scores[sorted_indices]
        
        print(f"Found {len(final_indices)} crops above similarity threshold {similarity_threshold}")
        
        if len(final_indices) == 0:
            print(f"No crops found above threshold {similarity_threshold}. Try lowering the threshold.")
            return 0, 0, 0, 0
        
    return crop_paths, final_indices, final_scores, similarity_threshold


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
        # Show top 40 results for display purposes
        final_indices = final_indices[:40]
        final_scores = final_scores[:40]
        rows, cols = 5, 8
        figsize = (16, 10)
        print(f"Showing top 40 results out of {n_results} matches")

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


def save_similar_results(crop_paths, final_indices, OUTPUT_DIR):
    for index in final_indices: 
        shutil.copy(crop_paths[index], OUTPUT_DIR)
