import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from threading import Lock
import uuid

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
# ✅ Use cosine similarity:
# If embeddings are normalized, cosine similarity = inner product
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

# ----------------------------
# LOAD OR CREATE METADATA
# ----------------------------
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata: List[Dict[str, Any]] = pickle.load(f)
else:
    metadata = []


# ----------------------------
# SAVE (SAFE)
# ----------------------------
def _atomic_write_bytes(path: str, data: bytes):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def _save():
    # Save FAISS
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata atomically
    meta_bytes = pickle.dumps(metadata)
    _atomic_write_bytes(METADATA_PATH, meta_bytes)


# ----------------------------
# SYNC CHECK
# ----------------------------
def _ensure_sync():
    """
    Ensure metadata length matches index.ntotal
    If mismatch happens due to crash, we truncate extra metadata.
    """
    ntotal = index.ntotal
    if len(metadata) > ntotal:
        del metadata[ntotal:]


# ----------------------------
# ADD EMBEDDINGS
# ----------------------------
def add_embeddings(
    embeddings: List[np.ndarray],
    image_url: str,
    event_id: str,
    bboxes: List[List[int]]
):
    """
    Adds face embeddings to FAISS and stores metadata per face.
    """
    if len(embeddings) != len(bboxes):
        raise ValueError("Embeddings and bboxes length mismatch")

    with lock:
        _ensure_sync()

        vectors = np.array(embeddings, dtype="float32")
        index.add(vectors)

        for bbox in bboxes:
            metadata.append({
                "id": uuid.uuid4().hex,
                "image_url": image_url,
                "event_id": event_id,
                "bbox": bbox
            })

        _save()


# ----------------------------
# SEARCH EMBEDDINGS
# ----------------------------
def search_embedding(
    embedding: np.ndarray,
    top_k: int = 50,
    similarity_threshold: float = 0.35
):
    """
    Cosine similarity based search:
    higher score = better match

    similarity_threshold:
      - 0.35 loose
      - 0.45 medium
      - 0.55 strict
    """
    with lock:
        _ensure_sync()

        if index.ntotal == 0:
            return []

        # FAISS expects shape (1, dim)
        query = np.array([embedding], dtype="float32")

        scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if idx >= len(metadata):
            continue

        if float(score) >= similarity_threshold:
            result = metadata[idx].copy()
            result["similarity"] = float(score)
            results.append(result)

    # Sort best first
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


# ----------------------------
# STATS
# ----------------------------
def stats():
    with lock:
        _ensure_sync()
        return {
            "faces_indexed": int(index.ntotal),
            "metadata_rows": int(len(metadata))
        }
