
# import os
# import faiss
# import pickle
# import numpy as np
# from typing import List, Dict
# from threading import Lock

# # ----------------------------
# # CONFIG
# # ----------------------------
# EMBEDDING_DIM = 512
# FAISS_INDEX_PATH = "faiss.index"
# METADATA_PATH = "metadata.pkl"

# lock = Lock()

# # ----------------------------
# # LOAD OR CREATE INDEX
# # ----------------------------
# if os.path.exists(FAISS_INDEX_PATH):
#     index = faiss.read_index(FAISS_INDEX_PATH)
# else:
#     index = faiss.IndexFlatL2(EMBEDDING_DIM)

# # ----------------------------
# # LOAD OR CREATE METADATA
# # ----------------------------
# if os.path.exists(METADATA_PATH):
#     with open(METADATA_PATH, "rb") as f:
#         metadata: List[Dict] = pickle.load(f)
# else:
#     metadata = []

# # ----------------------------
# # INTERNAL SAVE
# # ----------------------------
# def _save():
#     faiss.write_index(index, FAISS_INDEX_PATH)
#     with open(METADATA_PATH, "wb") as f:
#         pickle.dump(metadata, f)

# # ----------------------------
# # ADD EMBEDDINGS
# # ----------------------------
# def add_embeddings(
#     embeddings: List[np.ndarray],
#     image_url: str,
#     event_id: str = None,
#     user_id: str = None,
#     photo_id: str = None,
#     embedding_type: str = "photo"
# ):

#     with lock:

#         vectors = np.array(embeddings).astype("float32")

#         start_index = index.ntotal

#         index.add(vectors)

#         for i in range(len(embeddings)):

#             metadata.append({
#                 "faiss_index": start_index + i,
#                 "image_url": image_url,
#                 "event_id": event_id,
#                 "user_id": user_id,
#                 "photo_id": photo_id,
#                 "type": embedding_type
#             })

#         _save()

# # ----------------------------
# # STATS
# # ----------------------------
# def stats():
#     return {
#         "faces_indexed": index.ntotal
#     }

# # ----------------------------
# # SEARCH MATCHES FOR USER (FIXED)
# # ----------------------------
# # def search_user_matches(
# #     user_id: str,
# #     top_k: int = 50,
# #     distance_threshold: float = 2.0
# # ):

# def search_user_matches(
#     user_id: str,
#     top_k: int = 50,
#     distance_threshold: float = 0.9
# ):


#     print("SEARCHING MATCHES FOR USER:", user_id)

#     profile_indices = []

#     # Find profile embeddings for this user
#     for meta in metadata:

#         if meta.get("type") == "profile" and meta.get("user_id") == user_id:

#             profile_indices.append(meta["faiss_index"])

#     print("PROFILE FAISS INDICES:", profile_indices)

#     if not profile_indices:
#         print("NO PROFILE EMBEDDINGS FOUND")
#         return []

#     matches = []

#     # Search using each profile embedding
#     for faiss_idx in profile_indices:

#         profile_embedding = index.reconstruct(faiss_idx)

#         distances, indices = index.search(
#             np.array([profile_embedding]).astype("float32"),
#             top_k
#         )

        
            
            
#         for dist, match_idx in zip(distances[0], indices[0]):

#             if match_idx == -1:
#                 continue

#             if dist > distance_threshold:
#                 continue

#             match_meta = metadata[match_idx]

#             if match_meta.get("type") != "photo":
#                 continue

#     matches.append({
#         "image_url": match_meta.get("image_url"),
#         "distance": float(dist)
#     })
   

#     # Remove duplicates
#     unique = {}
#     for m in matches:
#         unique[m["image_url"]] = m

#     final_matches = list(unique.values())

#     print("TOTAL MATCHES RETURNED:", len(final_matches))

#     return final_matches



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
# Using cosine similarity (IndexFlatIP)
# ----------------------------
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

# ----------------------------
# LOAD OR CREATE METADATA
# ----------------------------
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "rb") as f:
        metadata: List[Dict] = pickle.load(f)
else:
    metadata = []

# ----------------------------
# INTERNAL SAVE
# ----------------------------
def _save():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

# ----------------------------
# NORMALIZE VECTOR
# ----------------------------
def normalize_vector(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ----------------------------
# ADD EMBEDDINGS
# ----------------------------
def add_embeddings(
    embeddings: List[np.ndarray],
    image_url: str,
    event_id: str = None,
    user_id: str = None,
    photo_id: str = None,
    embedding_type: str = "photo"
):

    with lock:

        # Normalize embeddings
        normalized_vectors = []

        for emb in embeddings:
            emb = emb.astype("float32")
            emb = normalize_vector(emb)
            normalized_vectors.append(emb)

        vectors = np.array(normalized_vectors).astype("float32")

        start_index = index.ntotal

        index.add(vectors)

        for i in range(len(normalized_vectors)):

            metadata.append({
                "faiss_index": start_index + i,
                "image_url": image_url,
                "event_id": event_id,
                "user_id": user_id,
                "photo_id": photo_id,
                "type": embedding_type
            })

        _save()

        print(f"Added {len(normalized_vectors)} embeddings")

# ----------------------------
# SEARCH MATCHES FOR USER
# Using cosine similarity
# ----------------------------
def search_user_matches(
    user_id: str,
    top_k: int = 50,
    similarity_threshold: float = 0.6
):

    print("SEARCHING MATCHES FOR USER:", user_id)

    profile_indices = []

    # Find profile embeddings
    for meta in metadata:

        if meta.get("type") == "profile" and meta.get("user_id") == user_id:

            profile_indices.append(meta["faiss_index"])

    print("PROFILE FAISS INDICES:", profile_indices)

    if not profile_indices:
        print("No profile embeddings found")
        return []

    matches = []

    for faiss_idx in profile_indices:

        profile_embedding = index.reconstruct(faiss_idx)

        profile_embedding = normalize_vector(profile_embedding)

        distances, indices = index.search(
            np.array([profile_embedding]).astype("float32"),
            top_k
        )

        for similarity, match_idx in zip(distances[0], indices[0]):

            if match_idx == -1:
                continue

            if similarity < similarity_threshold:
                continue

            match_meta = metadata[match_idx]

            if match_meta.get("type") != "photo":
                continue

            print(
                "MATCH FOUND:",
                match_meta["image_url"],
                "SIMILARITY:",
                similarity
            )

            matches.append({
                "image_url": match_meta["image_url"],
                "similarity": float(similarity)
            })

    # Remove duplicates
    unique = {}
    for m in matches:
        unique[m["image_url"]] = m

    final_matches = list(unique.values())

    print("TOTAL MATCHES RETURNED:", len(final_matches))

    return final_matches

# ----------------------------
# STATS
# ----------------------------
def stats():
    return {
        "faces_indexed": index.ntotal
    }

# ----------------------------
# REMOVE USER PROFILE EMBEDDINGS
# ----------------------------
def remove_user_profile(user_id: str):

    global metadata, index

    print("Removing old profile embeddings for user:", user_id)

    # Keep only non-profile or different user embeddings
    new_metadata = []
    new_vectors = []

    for i, meta in enumerate(metadata):

        if meta.get("type") == "profile" and meta.get("user_id") == user_id:
            continue

        new_metadata.append(meta)

        vec = index.reconstruct(meta["faiss_index"])
        new_vectors.append(vec)

    # Rebuild index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    if new_vectors:
        index.add(np.array(new_vectors).astype("float32"))

    # Update metadata with new indices
    for i in range(len(new_metadata)):
        new_metadata[i]["faiss_index"] = i

    metadata = new_metadata

    _save()

    print("Old profile embeddings removed")
