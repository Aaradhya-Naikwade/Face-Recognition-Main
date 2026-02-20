import cv2
import numpy as np
import insightface
from fastapi import HTTPException
from typing import List

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]  # change to CUDAExecutionProvider if GPU

# ----------------------------
# LOAD MODEL (ONCE)
# ----------------------------
face_app = insightface.app.FaceAnalysis(
    name=MODEL_NAME,
    providers=PROVIDERS
)
face_app.prepare(ctx_id=0)


# ----------------------------
# IMAGE UTIL
# ----------------------------
def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to OpenCV format
    """
    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    return img


# ----------------------------
# FACE EMBEDDING EXTRACTION
# ----------------------------
def extract_face_embeddings(image: np.ndarray) -> List[np.ndarray]:
    """
    Detect faces and return normalized embeddings
    """

    faces = face_app.get(image)

    if not faces:
        return []

    embeddings = []

    for face in faces:

        emb = face.embedding.astype("float32")

        # CRITICAL FIX: normalize embedding
        norm = np.linalg.norm(emb)

        if norm == 0:
            continue

        emb = emb / norm

        embeddings.append(emb)

    return embeddings
