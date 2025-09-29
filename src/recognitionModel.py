# Build embedding model (TensorFlow MobileNetV2) and compute embeddings
import numpy as np, os, json, glob, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.config import settings
from pathlib import Path
from tqdm import tqdm


def build_model_stream(IMG_SIZE: int, CROPS_DIR: str | os.PathLike):
    """
    Build embedding model and generate embeddings for crops in CROPS_DIR,
    using a streaming approach to avoid memory overload.
    """
    # MobileNetV2 backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation=None)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    embed_model = models.Model(inputs=base.input, outputs=x)

    # Gather crops recursively
    exts = [".jpg", ".jpeg", ".png"]
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
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
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

    # # Wrap generator in tf.data.Dataset for batching
    # ds = tf.data.Dataset.from_generator(
    #     gen,
    #     output_signature=tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
    # ).batch(64)

    # # Run embeddings without materializing the whole dataset
    # embs = embed_model.predict(ds, verbose=1)

    embs = embed_model.predict(X, batch_size=64, verbose=1)
    print(f"Generated embeddings of shape {embs.shape}")

    # Save
    np.save(settings.embed_path, embs)
    with open(settings.crop_path, "w") as f:
        json.dump(crop_paths, f)

    print(f"Saved embeddings to {settings.embed_path}")
    print(f"Saved crop paths to {settings.crop_path}")

    return embed_model

def build_model(IMG_SIZE, CROPS_DIR):
    """
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
    # Load MobileNetV2 pretrained on ImageNet, without top classification layers
    # include_top=False: Remove final classification layers
    # weights='imagenet': Use pretrained weights for better feature extraction
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
    
    # Custom head for embeddings
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layer: Reduce to 256-dimensional embedding space
    x = layers.Dense(256, activation=None)(x)
    
    # L2 normalization: cosine similarity and training stabilization
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    
    # Create the complete embedding model
    embed_model = models.Model(inputs=base.input, outputs=x)

    # Collect crops recursively (supporting subdirectories)
    crop_paths = sorted(
        glob.glob(os.path.join(CROPS_DIR, "**", "*.jpg"), recursive=True)
    )
    print(f"Found {len(crop_paths)} crops under {CROPS_DIR}")

    if len(crop_paths) == 0:
        print("No crops found.")
        return None
    
    # Helper to load and preprocess an image
    def load_img(path, IMG_SIZE):
        """
        Load and preprocess image for embedding generation.
        
        Steps:
        1. Load image and convert to RGB (removes alpha channel if present)
        2. Resize to model input size using high-quality BICUBIC interpolation
        3. Normalize pixel values to [0,1] range (neural network standard)
        """
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return np.asarray(img) / 255.0
    
    # Preprocess all crops
    X = np.stack([load_img(p, IMG_SIZE) for p in crop_paths], axis=0)

    # Generate embeddings
    embs = embed_model.predict(X, batch_size=64)

    # Save embeddings + paths
    np.save(settings.embed_path, embs)
    with open(settings.crop_path, "w") as f:
        json.dump(crop_paths, f)

    print(f"Saved embeddings to {settings.embed_path}") # (N, 256) array of feature vectors
    print(f"Saved crop paths to {settings.crop_path}") # List mapping array indices to file paths

    return embed_model


def char_nearest_neighbor(EMBED_PATH, CROP_PATH, IMG_SIZE, embed_model, seed_paths):
    """
    Find similar character crops using nearest neighbor search in embedding space.
    
    This function implements multi-seed similarity search:
    1. Load precomputed embeddings database
    2. Process seed images to create query embedding
    3. Find most similar crops using cosine distance
    4. Display results in a grid visualization
    
    Args:
        EMBED_PATH: Path to saved embeddings (.npy file)
        CROP_PATH: Path to saved crop paths (.json file) 
        IMG_SIZE: Image size for preprocessing seed images
        embed_model: Trained embedding model
        seed_paths: List of paths to seed/example character images
    """
    # Load precomputed embeddings database
    embs = np.load('data/embeddings.npy')
    with open('data/crop_paths.json','r') as f:
        crop_paths = json.load(f)

    # Initialize nearest neighbor search with cosine distance
    # n_neighbors=50: Prepare to find up to 50 similar images
    # metric='cosine': Use cosine similarity (1 - cosine_distance)
    # Cosine similarity works well with L2-normalized embeddings
    nn = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn.fit(embs)

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

        # Find nearest neighbors in embedding space
        # Returns: distances (similarity scores) and indices into crop database
        dists, idxs = nn.kneighbors(query_vec, n_neighbors=40)

        # Visualization: Display results in 5x8 grid
        plt.figure(figsize=(14, 10))
        for i, idx in enumerate(idxs[0]):
            # Load and resize crop for display
            im = Image.open(crop_paths[idx]).convert('RGB')
            plt.subplot(5, 8, i+1)
            plt.imshow(im.resize((128,128)))
            
            # Show cosine distance as title (lower = more similar)
            plt.title(f"{dists[0,i]:.3f}")
            plt.axis('off')
        plt.show()
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
    img = Image.open(path).convert('RGB').resize((size,size), Image.BICUBIC)
    return np.asarray(img)/255.0