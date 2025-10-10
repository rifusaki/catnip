import numpy as np, json
from sklearn.linear_model import LogisticRegression
from .embeddingModel import load_img
from src.config import settings

def izutsumi_query(EMBED_PATH, CROP_PATH, IMG_SIZE, embed_model, seed_paths, neg_paths, similarity_threshold=0.7, alpha=0.5, mode='max'):
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
        seed_paths: List of paths to seeds of Izutsumi
        neg_path: List of paths to seeds of Anti-Izutsumi
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
        # Process positive and negative seeds to create query embedding
        query_vec, neg_vec = _helper_embeddings(IMG_SIZE, embed_model, seed_paths, neg_paths)

        # Calculate cosine similarities with all embeddings
        similarity = np.dot(embs, query_vec.T)
        neg_similarity = np.dot(embs, neg_vec.T)
        
        # Filter by similarity threshold
        if mode == "max":
            pos_score = similarity.max(axis=1)
            neg_score = neg_similarity.max(axis=1) if neg_similarity is not None else 0
        else:  # mean
            pos_score = similarity.mean(axis=1)
            neg_score = neg_similarity.mean(axis=1) if neg_similarity is not None else 0

        score = pos_score - neg_score * alpha

        similar_indices = np.where(score >= similarity_threshold)[0]
        similar_scores = score[similar_indices]
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similar_scores)[::-1]
        final_indices = similar_indices[sorted_indices]
        final_scores = similar_scores[sorted_indices]
        
        print(f"Found {len(final_indices)} crops above similarity threshold {similarity_threshold}")
        
        if len(final_indices) == 0:
            print(f"No crops found above threshold {similarity_threshold}. Try lowering the threshold.")
            return 0, 0, 0, 0
        
    return crop_paths, final_indices, final_scores, similarity_threshold


def _helper_embeddings(IMG_SIZE, embed_model, seed_paths, neg_paths):
    # Process seed images to create query embedding
    X_seed = np.stack([load_img(p, IMG_SIZE) for p in seed_paths], axis=0)
    seed_embs = embed_model.predict(X_seed, batch_size=8, verbose=0)

    neg_seed = np.stack([load_img(p, IMG_SIZE) for p in neg_paths], axis=0)
    neg_embs = embed_model.predict(neg_seed, batch_size=8, verbose=0)

    # Average multiple seed embeddings for robust query representation
    query_vec = np.mean(seed_embs, axis=0, keepdims=True)
    neg_vec = np.mean(neg_embs, axis=0, keepdims=True)

    # Re-normalize after averaging (maintains unit length property)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    neg_vec = neg_vec / np.linalg.norm(neg_vec, axis=1, keepdims=True)

    return query_vec, neg_vec


def izutsuminess_rank(EMBED_PATH, CROP_PATH, embed_model, seed_paths, neg_paths):
    """
    Rank crops based on logistic regression.

    Returns:
        ranked_idx: indices to sort crops[ranked_idx] by likelihood of being Izutsumi
    """
    with open(CROP_PATH, 'r') as f:
        crop_paths = json.load(f)

    X_pos, X_neg = _helper_embeddings(settings.img_size, embed_model, seed_paths, neg_paths)
    embs = np.load(EMBED_PATH)

    # y: labels (1 for Izutsumi, 0 for not)
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X, y)

    # Predict probability for all 20k crops
    probs = clf.predict_proba(embs)[:, 1]  # probability of being Izutsumi
    ranked_idx = np.argsort(-probs)

    return ranked_idx

