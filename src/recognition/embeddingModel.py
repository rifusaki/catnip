# Build embedding model (TensorFlow MobileNetV2) and compute embeddings
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.config import settings
from pathlib import Path
from tqdm import tqdm

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