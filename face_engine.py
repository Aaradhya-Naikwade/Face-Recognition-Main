
import cv2
import numpy as np
import insightface
from fastapi import HTTPException
from typing import List, Dict

MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]

face_app = insightface.app.FaceAnalysis(
    name=MODEL_NAME,
    providers=PROVIDERS
)
face_app.prepare(ctx_id=0)

def decode_image(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        raise HTTPException(400, "Invalid or corrupted image")
    return img

def extract_face_data(image: np.ndarray) -> List[Dict]:
    faces = face_app.get(image)
    if not faces:
        return []

    # Prefer largest face (very important for search accuracy)
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )

    results = []

    # ONLY largest face for search
    for face in faces[:1]:
        emb = face.embedding.astype("float32")
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        emb = emb / norm

        bbox = face.bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox

        # Ignore very small faces
        if (x2 - x1) * (y2 - y1) < 1600:
            continue

        results.append({
            "embedding": emb,
            "bbox": bbox
        })

    return results
