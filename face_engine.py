import os
import cv2
import numpy as np
from fastapi import HTTPException
from typing import List

# ----------------------------
# FORCE MODEL CACHE OUTSIDE PROJECT
# ----------------------------
os.environ["INSIGHTFACE_HOME"] = "/opt/render/.insightface"

import insightface
from insightface.app import FaceAnalysis

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "buffalo_s"
PROVIDERS = ["CPUExecutionProvider"]

# ----------------------------
# LAZY MODEL LOADING
# ----------------------------
_face_app = None

def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name=MODEL_NAME,
            providers=PROVIDERS
        )
        _face_app.prepare(ctx_id=0)
    return _face_app


# ----------------------------
# IMAGE UTIL
# ----------------------------
def decode_image(image_bytes: bytes) -> np.ndarray:
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
    face_app = get_face_app()
    faces = face_app.get(image)

    if not faces:
        return []

    return [face.embedding for face in faces]
