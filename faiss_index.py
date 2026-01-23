import os
import faiss
import pickle
import numpy as np
from typing import List, Dict
from threading import Lock

# ----------------------------
# CONFIG
# ----------------------------
EMBEDDING_DIM = 512
FAISS_INDEX_PATH = "faiss.index"
METADATA_PATH = "metadata.pkl"

lock = Lock()

# ----------------------------
# LOAD OR CREATE INDEX
# ----------------------------
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

# ----------------------------
# LOAD OR CREATE METADATA
# ----------------------------
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata: List[Dict] = pickle.load(f)
else:
    metadata = []

# ----------------------------
# SAVE
# ----------------------------
def _save():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

# ----------------------------
# ADD EMBEDDINGS
# ----------------------------
def add_embeddings(
    embeddings: List[np.ndarray],
    image_url: str,
    event_id: str
):
    with lock:
        vectors = np.array(embeddings).astype("float32")
        index.add(vectors)

        for _ in embeddings:
            metadata.append({
                "image_url": image_url,
                "event_id": event_id
            })

        _save()

# ----------------------------
# SEARCH EMBEDDINGS
# ----------------------------
def search_embedding(
    embedding: np.ndarray,
    top_k: int = 50,
    distance_threshold: float = 1.4
):
    with lock:
        if index.ntotal == 0:
            return []

        distances, indices = index.search(
            np.array([embedding]).astype("float32"),
            top_k
        )

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        if dist <= distance_threshold:
            result = metadata[idx].copy()
            result["distance"] = float(dist)
            results.append(result)

    return results

# ----------------------------
# STATS
# ----------------------------
def stats():
    return {
        "faces_indexed": index.ntotal
    }
