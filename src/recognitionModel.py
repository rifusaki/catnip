# Build embedding model (TensorFlow MobileNetV2) and compute embeddings
import numpy as np, os, json, glob, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.config import settings
from pathlib import Path
from tqdm import tqdm


def build_model(IMG_SIZE: int, CROPS_DIR: str | os.PathLike):
    """
    Build embedding model and generate embeddings for crops in CROPS_DIR,
    using a streaming approach to avoid memory overload.

    Build a feature extraction model using MobileNetV2 and compute embeddings for all crops.
    
    Architecture:
    1. MobileNetV2 backbone (ImageNet pretrained) - removes top classification layers
    2. Global Average Pooling - converts feature maps to single vector per image
    3. Dense layer (256 neurons) - reduces dimensionality 
    4. L2 normalization - ensures all embeddings have unit length for cosine similarity
    
    Args:
        IMG_SIZE: Input image size (crops will be resized to IMG_SIZE x IMG_SIZE)
        CROPS_DIR: Directory containing character crop images
        
    Returns:
        embed_model: Trained embedding model for generating features
    """
    # MobileNetV2 backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=None)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    embed_model = models.Model(inputs=base.input, outputs=x)

    # Gather crops recursively
    crop_paths = [
        str(p) for p in Path(CROPS_DIR).rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Found {len(crop_paths)} crops under {CROPS_DIR}")

    if not crop_paths:
        return None

    # Generator: yields one image at a time
    def gen():
        for path in crop_paths:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
            yield np.asarray(img) / 255.0

    X, valid_paths = [], []
    for p in tqdm(crop_paths, desc="Loading crops"):
        try:
            arr = load_img(p, IMG_SIZE)
            X.append(arr)
            valid_paths.append(str(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    
    X = np.stack(X, axis=0)
    print(f"Successfully loaded {len(valid_paths)} crops")

    embs = embed_model.predict(X, batch_size=64, verbose=1)
    print(f"Generated embeddings of shape {embs.shape}")

    # Save
    np.save(settings.embed_path, embs)
    with open(settings.crop_path, "w") as f:
        json.dump(crop_paths, f)

    print(f"Saved embeddings to {settings.embed_path}")
    print(f"Saved crop paths to {settings.crop_path}")

    return embed_model


def char_nearest_neighbor(EMBED_PATH, CROP_PATH, IMG_SIZE, embed_model, seed_paths, similarity_threshold=0.7):
    """
    Find similar character crops using similarity threshold in embedding space.
    
    This function implements threshold-based similarity search:
    1. Load precomputed embeddings database
    2. Process seed images to create query embedding
    3. Find all crops above similarity threshold using cosine distance
    4. Display results sorted by similarity score
    
    Args:
        EMBED_PATH: Path to saved embeddings (.npy file)
        CROP_PATH: Path to saved crop paths (.json file) 
        IMG_SIZE: Image size for preprocessing seed images
        embed_model: Trained embedding model
        seed_paths: List of paths to seed/example character images
        similarity_threshold: Minimum cosine similarity (0-1, higher = more similar)
    """
    # Load precomputed embeddings database
    embs = np.load('data/embeddings.npy')
    with open('data/crop_paths.json','r') as f:
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
            return
        
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
        
        return final_indices, final_scores
    else:
        print("Add paths of seeds.")

# Multiple seeds - standalone image loading utility
def load_img(path, size):
    """
    Utility function for loading and preprocessing individual images.
    
    Used by char_nearest_neighbor for processing seed images.
    Identical to the nested function in build_model but available globally.
    
    Args:
        path: Path to image file
        size: Target size (will create size x size square image)
        
    Returns:
        Normalized numpy array ready for model input
    """
    img = Image.open(path).convert('RGB').resize((size,size), Image.Resampling.BICUBIC)
    return np.asarray(img)/255.0