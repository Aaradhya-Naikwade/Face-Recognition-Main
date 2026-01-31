

# import os
# import faiss
# import pickle
# import numpy as np
# from typing import List, Dict, Any
# from threading import Lock
# import uuid

# EMBEDDING_DIM = 512
# FAISS_INDEX_PATH = "faiss.index"
# METADATA_PATH = "metadata.pkl"

# lock = Lock()

# if os.path.exists(FAISS_INDEX_PATH):
#     index = faiss.read_index(FAISS_INDEX_PATH)
# else:
#     index = faiss.IndexFlatIP(EMBEDDING_DIM)

# if os.path.exists(METADATA_PATH):
#     with open(METADATA_PATH, "rb") as f:
#         metadata: List[Dict[str, Any]] = pickle.load(f)
# else:
#     metadata = []

# def _save():
#     faiss.write_index(index, FAISS_INDEX_PATH)
#     with open(METADATA_PATH, "wb") as f:
#         pickle.dump(metadata, f)

# def add_embeddings(
#     embeddings: List[np.ndarray],
#     image_url: str,
#     group_id: str,
#     photographer_id: str,
#     bboxes: List[List[int]]
# ):
#     with lock:
#         vectors = np.array(embeddings, dtype="float32")
#         faiss.normalize_L2(vectors)
#         index.add(vectors)

#         for bbox in bboxes:
#             metadata.append({
#                 "id": uuid.uuid4().hex,
#                 "image_url": image_url,
#                 "group_id": group_id,
#                 "photographer_id": photographer_id,
#                 "bbox": bbox,
#                 "deleted": False
#             })

#         _save()

# def search_embedding(
#     embedding: np.ndarray,
#     top_k: int,
#     similarity_threshold: float
# ):
#     with lock:
#         if index.ntotal == 0:
#             return []

#         query = np.array([embedding], dtype="float32")
#         faiss.normalize_L2(query)
#         scores, indices = index.search(query, top_k)

#     results = []
#     for score, idx in zip(scores[0], indices[0]):
#         if idx == -1 or idx >= len(metadata):
#             continue

#         m = metadata[idx]
#         if m.get("deleted"):
#             continue

#         if score >= similarity_threshold:
#             r = m.copy()
#             r["similarity"] = float(score)
#             results.append(r)

#     return sorted(results, key=lambda x: x["similarity"], reverse=True)

# def stats():
#     return {
#         "faces_indexed": index.ntotal,
#         "metadata_rows": len(metadata)
#     }


















import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from threading import Lock
import uuid

EMBEDDING_DIM = 512
FAISS_INDEX_PATH = "faiss.index"
METADATA_PATH = "metadata.pkl"

lock = Lock()

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata: List[Dict[str, Any]] = pickle.load(f)
else:
    metadata = []

def _save():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def add_embeddings(
    embeddings: List[np.ndarray],
    image_url: str,
    event_id: str,          
    group_id: str,
    photographer_id: str,
    bboxes: List[List[int]]
):
    with lock:
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)

        for bbox in bboxes:
            metadata.append({
                "id": uuid.uuid4().hex,
                "image_url": image_url,
                "event_id": event_id,          
                "group_id": group_id,
                "photographer_id": photographer_id,
                "bbox": bbox,
                "deleted": False
            })

        _save()


def search_embedding(
    embedding: np.ndarray,
    top_k: int,
    similarity_threshold: float
):
    with lock:
        if index.ntotal == 0:
            return []

        query = np.array([embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        m = metadata[idx]
        if m.get("deleted"):
            continue

        if score >= similarity_threshold:
            r = m.copy()
            r["similarity"] = float(score)
            results.append(r)

    return sorted(results, key=lambda x: x["similarity"], reverse=True)

def stats():
    return {
        "faces_indexed": index.ntotal,
        "metadata_rows": len(metadata)
    }
