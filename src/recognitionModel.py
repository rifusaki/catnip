# Build embedding model (TensorFlow MobileNetV2) and compute embeddings
import numpy as np, os, json, glob, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def build_model(IMG_SIZE, CROPS_DIR):
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation=None)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    embed_model = models.Model(inputs=base.input, outputs=x)

    crop_paths = sorted(glob.glob(os.path.join(CROPS_DIR, '*.jpg')))
    print('Found', len(crop_paths), 'crops')
    if len(crop_paths) > 0:
        def load_img(path, IMG_SIZE):
            img = Image.open(path).convert('RGB').resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)
            return np.asarray(img)/255.0
        X = np.stack([load_img(p, IMG_SIZE) for p in crop_paths], axis=0)
        embs = embed_model.predict(X, batch_size=64)
        np.save('data/embeddings.npy', embs)
        with open('data/crop_paths.json', 'w') as f:
            json.dump(crop_paths, f)
        print('Saved embeddings and paths')
        return embed_model
    else:
        print('No crops found.')


def char_nearest_neighbor(EMBED_PATH, CROP_PATH, IMG_SIZE, embed_model, seed_paths):
    embs = np.load('data/embeddings.npy')
    with open('data/crop_paths.json','r') as f:
        crop_paths = json.load(f)

    nn = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn.fit(embs)

    if len(seed_paths) > 0:
        X_seed = np.stack([load_img(p, IMG_SIZE) for p in seed_paths], axis=0)
        seed_embs = embed_model.predict(X_seed, batch_size=8)
        query_vec = np.mean(seed_embs, axis=0, keepdims=True)  # average embedding
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Nearest neighbors
        dists, idxs = nn.kneighbors(query_vec, n_neighbors=40)

        plt.figure(figsize=(14, 10))
        for i, idx in enumerate(idxs[0]):
            im = Image.open(crop_paths[idx]).convert('RGB')
            plt.subplot(5, 8, i+1)
            plt.imshow(im.resize((128,128)))
            plt.title(f"{dists[0,i]:.3f}")
            plt.axis('off')
        plt.show()
    else:
        print("Add paths of seeds.")

# Multiple seeds
def load_img(path, size):
    img = Image.open(path).convert('RGB').resize((size,size), Image.BICUBIC)
    return np.asarray(img)/255.0