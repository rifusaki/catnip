# Build embedding model (TensorFlow MobileNetV2) and compute embeddings
import numpy as np, os, json, glob, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.config import settings
from pathlib import Path
from tqdm import tqdm


def build_model(IMG_SIZE: int, CROPS_DIR: str | os.PathLike, load_weights: bool = True):
    """
    Build embedding model and optionally load saved weights for consistency.

    Build a feature extraction model using MobileNetV2 and compute embeddings for all crops.
    
    Architecture:
    1. MobileNetV2 backbone (ImageNet pretrained) - removes top classification layers
    2. Global Average Pooling - converts feature maps to single vector per image
    3. Dense layer - reduces dimensionality 
    4. L2 normalization - all embeddings have unit length for cosine similarity
    
    Args:
        IMG_SIZE: Input image size (crops will be resized to IMG_SIZE x IMG_SIZE)
        CROPS_DIR: Directory containing character crop images
        load_weights: If True, try to load saved model weights for consistency
        
    Returns:
        embed_model: Trained embedding model for generating features
    """
    # MobileNet backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=None)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    embed_model = models.Model(inputs=base.input, outputs=x)
    
    # Try to load saved weights if they exist and load_weights=True
    if load_weights:
        model_weights_path = settings.embed_path.parent / "model.weights.h5"
        if model_weights_path.exists():
            try:
                embed_model.load_weights(str(model_weights_path))
                print(f"Loaded model weights from {model_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using fresh model weights (may cause embedding mismatch)")

    return embed_model


def compute_embeddings(embed_model, CROPS_DIR: str | os.PathLike, IMG_SIZE: int):
    """
    Compute embeddings for all crops in CROPS_DIR using build_model and save them.
    """
    # Gather crops recursively
    crop_paths = [
        str(p) for p in Path(CROPS_DIR).rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Found {len(crop_paths)} crops under {CROPS_DIR}")

    if not crop_paths:
        return None

    def batch_generator(paths, batch_size):
        batch, batch_paths = [], []
        for p in paths:
            try:
                arr = load_img(p, IMG_SIZE)
                batch.append(arr)
                batch_paths.append(str(p))
            except Exception as e:
                print(f"Skipping {p}: {e}")
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0), batch_paths
                batch, batch_paths = [], []
        if batch:
            yield np.stack(batch, axis=0), batch_paths

    batch_size = 64
    all_embs, valid_paths = [], []
    for X_batch, paths_batch in tqdm(batch_generator(crop_paths, batch_size), total=(len(crop_paths) + batch_size - 1) // batch_size, desc="Embedding batches"):
        embs_batch = embed_model.predict(X_batch, batch_size=batch_size, verbose=0)
        all_embs.append(embs_batch)
        valid_paths.extend(paths_batch)

    embs = np.concatenate(all_embs, axis=0)
    print(f"Successfully embedded {len(valid_paths)} crops")
    print(f"Generated embeddings of shape {embs.shape}")

    # Save embeddings, crop paths, and model weights
    np.save(settings.embed_path, embs)
    with open(settings.crop_path, "w") as f:
        json.dump(valid_paths, f)
    
    # Save model weights for consistency across sessions
    model_weights_path = settings.model_dir / "model.weights.h5"
    embed_model.save_weights(str(model_weights_path))

    print(f"Saved embeddings to {settings.embed_path}")
    print(f"Saved crop paths to {settings.crop_path}")
    print(f"Saved model weights to {model_weights_path}")

    return embs, valid_paths


def load_embeddings(EMBED_PATH = settings.embed_path, CROP_PATH = settings.crop_path):
    """
    Load embeddings and crop paths.
    """
    embs = np.load(EMBED_PATH)
    with open(CROP_PATH, "r") as f:
        crop_paths = json.load(f)

    print(f"Loaded embeddings of shape {embs.shape}")

    return embs, crop_paths


# Multiple seeds - standalone image loading utility
def load_img(path, size):
    """
    Utility function for loading and preprocessing individual images.
    
    Args:
        path: Path to image file
        size: Target size (will create size x size square image)
        
    Returns:
        Normalized numpy array ready for model input
    """
    img = Image.open(path).convert('RGB').resize((size,size), Image.Resampling.BICUBIC)
    return np.asarray(img)/255.00