# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional

# from face_engine import decode_image, extract_face_embeddings
# from faiss_index import add_embeddings, search_embedding, stats

# # ----------------------------
# # CONFIG
# # ----------------------------
# DISTANCE_THRESHOLD = 1.0

# # ----------------------------
# # APP INIT
# # ----------------------------
# app = FastAPI(
#     title="Face Recognition API",
#     version="1.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------
# # ADD EVENT IMAGE
# # ----------------------------
# @app.post("/add-face")
# async def add_face(
#     file: UploadFile = File(...),
#     image_url: str = Form(...),
#     event_id: Optional[str] = Form(None)
# ):
#     image_bytes = await file.read()
#     image = decode_image(image_bytes)

#     embeddings = extract_face_embeddings(image)
#     if not embeddings:
#         raise HTTPException(status_code=400, detail="No face detected")

#     add_embeddings(
#         embeddings=embeddings,
#         image_url=image_url,
#         event_id=event_id
#     )

#     return {
#         "status": "success",
#         "faces_added": len(embeddings)
#     }

# # ----------------------------
# # SEARCH SELFIE
# # ----------------------------
# @app.post("/search-face")
# async def search_face(
#     file: UploadFile = File(...),
#     top_k: int = 5
# ):
#     image_bytes = await file.read()
#     image = decode_image(image_bytes)

#     embeddings = extract_face_embeddings(image)
#     if not embeddings:
#         raise HTTPException(status_code=400, detail="No face detected")

#     matches = []
#     for emb in embeddings:
#         matches.extend(
#             search_embedding(
#                 embedding=emb,
#                 top_k=top_k,
#                 distance_threshold=DISTANCE_THRESHOLD
#             )
#         )

#     return {
#         "matches": matches
#     }

# # ----------------------------
# # HEALTH CHECK
# # ----------------------------
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         **stats()
#     }







from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from face_engine import decode_image, extract_face_embeddings
from faiss_index import (
    add_embeddings,
    search_user_matches,
    stats
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODELS
# -------------------------

class RegisterFaceRequest(BaseModel):
    user_id: str
    image_url: str


class IndexPhotoRequest(BaseModel):
    photo_id: str
    photo_url: str
    group_id: str


class FindMatchesRequest(BaseModel):
    user_id: str


# -------------------------
# HELPER
# -------------------------

def download_image(url: str):

    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(400, "Cannot download image")

    return decode_image(response.content)


# -------------------------
# REGISTER PROFILE FACE
# -------------------------

# @app.post("/register-face")
# def register_face(req: RegisterFaceRequest):

#     image = download_image(req.image_url)

#     embeddings = extract_face_embeddings(image)

#     if not embeddings:
#         raise HTTPException(400, "No face detected")

#     add_embeddings(
#         embeddings=embeddings,
#         image_url=req.image_url,
#         user_id=req.user_id,
#         embedding_type="profile"
#     )

#     return {"status": "ok"}

@app.post("/register-face")
def register_face(req: RegisterFaceRequest):

    from faiss_index import remove_user_profile

    # REMOVE OLD PROFILE FIRST
    remove_user_profile(req.user_id)

    image = download_image(req.image_url)

    embeddings = extract_face_embeddings(image)

    if not embeddings:
        raise HTTPException(400, "No face detected")

    add_embeddings(
        embeddings=embeddings,
        image_url=req.image_url,
        user_id=req.user_id,
        embedding_type="profile"
    )

    return {"status": "ok"}


# -------------------------
# INDEX GROUP PHOTO
# -------------------------

@app.post("/index-group-photo")
def index_group_photo(req: IndexPhotoRequest):

    image = download_image(req.photo_url)

    embeddings = extract_face_embeddings(image)

    if not embeddings:
        return {"status": "no faces"}

    add_embeddings(
        embeddings=embeddings,
        image_url=req.photo_url,
        event_id=req.group_id,
        photo_id=req.photo_id,
        embedding_type="photo"
    )
    

    return {"status": "ok"}


# -------------------------
# FIND MATCHES
# -------------------------

# @app.post("/find-matches")
# def find_matches(req: FindMatchesRequest):

#     matches = search_user_matches(req.user_id)

#     return {
#         "matches": matches
#     }


@app.post("/find-matches")
def find_matches(req: FindMatchesRequest):

    print("FIND MATCHES CALLED")
    print("USER ID:", req.user_id)

    from faiss_index import metadata

    print("TOTAL METADATA:", len(metadata))

    profile_count = 0
    photo_count = 0

    for m in metadata:
        if m.get("type") == "profile":
            profile_count += 1
        if m.get("type") == "photo":
            photo_count += 1
        
    print("PROFILE EMBEDDINGS:", profile_count)
    print("PHOTO EMBEDDINGS:", photo_count)

    # matches = search_user_matches(req.user_id, distance_threshold=2.0)
    matches = search_user_matches(req.user_id, similarity_threshold=0.45)


    print("MATCHES FOUND:", len(matches))

    return {
        "matches": matches
    }


# -------------------------
# HEALTH
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        **stats()
    }
