
    
# from dotenv import load_dotenv
# load_dotenv()

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import os
# from urllib.parse import urlparse
# import boto3

# from face_engine import decode_image, extract_face_data
# from faiss_index import add_embeddings, search_embedding, metadata, _save, lock
# from s3_client import upload_image_to_s3

# # --------------------------------------------------
# # CONFIG
# # --------------------------------------------------

# # 🔑 Keep Python threshold LOWER
# # Next.js will apply stricter filtering
# SIMILARITY_THRESHOLD = 0.45

# app = FastAPI(title="KlickShare Face Service")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------------------------------------------------
# # HELPERS
# # --------------------------------------------------

# def ensure_min_size(image: np.ndarray, min_size=640) -> np.ndarray:
#     """Resize image if too small (critical for InsightFace)"""
#     h, w = image.shape[:2]
#     if max(h, w) < min_size:
#         scale = min_size / max(h, w)
#         image = cv2.resize(image, (int(w * scale), int(h * scale)))
#     return image

# # --------------------------------------------------
# # ADD FACE (Photographer Upload)
# # --------------------------------------------------
# @app.post("/add-face")
# async def add_face(
#     file: UploadFile = File(...),
#     group_id: str = Form(...),
#     photographer_id: str = Form(...)
# ):
#     image_bytes = await file.read()
#     image = decode_image(image_bytes)
#     image = ensure_min_size(image)

#     face_data = extract_face_data(image)

#     # ❌ DO NOT HARD FAIL
#     if not face_data:
#         return {
#             "status": "no_face_detected",
#             "faces_indexed": 0
#         }

#     # Upload to S3 ONCE
#     s3_url = upload_image_to_s3(
#         image_bytes=image_bytes,
#         group_id=group_id,
#         filename=file.filename
#     )

#     embeddings = [f["embedding"] for f in face_data]
#     bboxes = [f["bbox"] for f in face_data]

#     add_embeddings(
#         embeddings=embeddings,
#         image_url=s3_url,
#         group_id=group_id,
#         photographer_id=photographer_id,
#         bboxes=bboxes
#     )

#     return {
#         "status": "success",
#         "s3_url": s3_url,
#         "faces_indexed": len(embeddings)
#     }

# # --------------------------------------------------
# # SEARCH FACE (User Selfie)
# # --------------------------------------------------

# @app.post("/search-face")
# async def search_face(
#     file: UploadFile = File(...),
#     top_k: int = 50
# ):
#     image_bytes = await file.read()
#     image = decode_image(image_bytes)
#     image = ensure_min_size(image)

#     face_data = extract_face_data(image)

#     # ✅ Soft failure (never throw)
#     if not face_data:
#         return {
#             "matches": [],
#             "warning": "no_face_detected",
#             "faces_detected": 0
#     }


#     results = []

#     for f in face_data:
#         results.extend(
#             search_embedding(
#                 embedding=f["embedding"],
#                 top_k=50,
#                 similarity_threshold=SIMILARITY_THRESHOLD
#             )
#         )

#     # Deduplicate by face-id
#     unique = {}
#     for r in results:
#         unique[r["id"]] = r

#     return {
#         "matches": list(unique.values()),
#         "threshold": SIMILARITY_THRESHOLD
#     }

# # --------------------------------------------------
# # DELETE IMAGE (S3 + FAISS)
# # --------------------------------------------------
# @app.post("/delete-image")
# def delete_image(payload: dict):
#     image_url = payload.get("image_url")
#     if not image_url:
#         raise HTTPException(status_code=400, detail="image_url required")

#     # Remove from FAISS metadata
#     with lock:
#         before = len(metadata)
#         metadata[:] = [m for m in metadata if m.get("image_url") != image_url]
#         if len(metadata) != before:
#             _save()

#     # Delete from S3
#     parsed = urlparse(image_url)
#     s3_key = parsed.path.lstrip("/")

#     s3 = boto3.client(
#         "s3",
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name=os.getenv("AWS_REGION"),
#     )

#     s3.delete_object(Bucket="haltn", Key=s3_key)

#     return {
#         "status": "deleted",
#         "image_url": image_url
#     }











    
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from urllib.parse import urlparse
import boto3

from face_engine import decode_image, extract_face_data
from faiss_index import add_embeddings, search_embedding, metadata, _save, lock
from s3_client import upload_image_to_s3

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# 🔑 Keep Python threshold LOWER
# Next.js will apply stricter filtering
SIMILARITY_THRESHOLD = 0.45

app = FastAPI(title="KlickShare Face Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def ensure_min_size(image: np.ndarray, min_size=640) -> np.ndarray:
    """Resize image if too small (critical for InsightFace)"""
    h, w = image.shape[:2]
    if max(h, w) < min_size:
        scale = min_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# --------------------------------------------------
# ADD FACE (Photographer Upload)
# --------------------------------------------------
@app.post("/add-face")
async def add_face(
    file: UploadFile = File(...),
    event_id: str = Form(...),          # ✅ ADD THIS
    group_id: str = Form(...),
    photographer_id: str = Form(...)
):
    image_bytes = await file.read()

    image = decode_image(image_bytes)
    face_data = extract_face_data(image)

    if not face_data:
        raise HTTPException(400, "No face detected")

    embeddings = [f["embedding"] for f in face_data]
    bboxes = [f["bbox"] for f in face_data]

    # ✅ FIXED CALL (event_id added)
    s3_url = upload_image_to_s3(
        image_bytes=image_bytes,
        event_id=event_id,
        group_id=group_id,
        filename=file.filename
    )

    add_embeddings(
        embeddings=embeddings,
        image_url=s3_url,
        event_id=event_id,             
        group_id=group_id,
        photographer_id=photographer_id,
        bboxes=bboxes
    )

    return {"s3_url": s3_url}


# --------------------------------------------------
# SEARCH FACE (User Selfie)
# --------------------------------------------------

@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = 50
):
    image_bytes = await file.read()
    image = decode_image(image_bytes)
    image = ensure_min_size(image)

    face_data = extract_face_data(image)

    # ✅ Soft failure (never throw)
    if not face_data:
        return {
            "matches": [],
            "warning": "no_face_detected",
            "faces_detected": 0
    }


    results = []

    for f in face_data:
        results.extend(
            search_embedding(
                embedding=f["embedding"],
                top_k=50,
                similarity_threshold=SIMILARITY_THRESHOLD
            )
        )

    # Deduplicate by face-id
    unique = {}
    for r in results:
        unique[r["id"]] = r

    return {
        "matches": list(unique.values()),
        "threshold": SIMILARITY_THRESHOLD
    }

# --------------------------------------------------
# DELETE IMAGE (S3 + FAISS)
# --------------------------------------------------
@app.post("/delete-image")
def delete_image(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    # Remove from FAISS metadata
    with lock:
        before = len(metadata)
        metadata[:] = [m for m in metadata if m.get("image_url") != image_url]
        if len(metadata) != before:
            _save()

    # Delete from S3
    parsed = urlparse(image_url)
    s3_key = parsed.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    s3.delete_object(Bucket="haltn", Key=s3_key)

    return {
        "status": "deleted",
        "image_url": image_url
    }
