import os
import re
import uuid
from typing import Dict, List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

from face_engine import decode_image, extract_face_data
from faiss_index import (
    add_embeddings,
    search_embedding,
    stats
)

# =====================================================
# CONFIG
# =====================================================
UPLOAD_DIR = "uploads"

# Cosine similarity threshold
# loose: 0.35, medium: 0.45, strict: 0.55
SIMILARITY_THRESHOLD = 0.45


# =====================================================
# HELPERS
# =====================================================
def sanitize_event_id(event_id: str) -> str:
    """
    Prevent path traversal and weird folder names.
    Allowed: letters, numbers, underscore, dash
    """
    event_id = event_id.strip()
    if not event_id:
        raise HTTPException(status_code=400, detail="event_id is required")

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", event_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid event_id. Use only letters, numbers, _ or - (max 50 chars)."
        )
    return event_id


def is_allowed_image(filename: str) -> bool:
    filename = filename.lower()
    return filename.endswith((".jpg", ".jpeg", ".png", ".webp"))


# =====================================================
# APP INIT
# =====================================================
app = FastAPI(title="Face Recognition System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# STATIC FILES
# =====================================================
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# =====================================================
# UI ROUTES
# =====================================================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/ui/add-face")
def add_face_ui(request: Request):
    return templates.TemplateResponse(
        "add_face.html",
        {"request": request}
    )


@app.get("/ui/search")
def search_face_ui(request: Request):
    return templates.TemplateResponse(
        "search_face.html",
        {
            "request": request,
            "results": None
        }
    )


# =====================================================
# ADD FACE (API)
# =====================================================
@app.post("/add-face")
async def add_face(
    file: UploadFile = File(...),
    event_id: str = Form(...)
):
    event_id = sanitize_event_id(event_id)

    if not is_allowed_image(file.filename):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    contents = await file.read()

    event_folder = os.path.join(UPLOAD_DIR, event_id)
    os.makedirs(event_folder, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(event_folder, unique_name)

    with open(file_path, "wb") as f:
        f.write(contents)

    image = decode_image(contents)

    face_data = extract_face_data(image)
    if not face_data:
        raise HTTPException(status_code=400, detail="No face detected")

    embeddings = [fd["embedding"] for fd in face_data]
    bboxes = [fd["bbox"] for fd in face_data]

    image_url = f"/uploads/{event_id}/{unique_name}"

    add_embeddings(
        embeddings=embeddings,
        image_url=image_url,
        event_id=event_id,
        bboxes=bboxes
    )

    return {
        "status": "success",
        "faces_added": len(embeddings),
        "image_url": image_url,
        "event_id": event_id
    }


# =====================================================
# ADD FACE (UI POST)
# =====================================================
@app.post("/ui/add-face")
async def add_face_ui_post(
    request: Request,
    file: UploadFile = File(...),
    event_id: str = Form(...)
):
    try:
        event_id = sanitize_event_id(event_id)

        if not is_allowed_image(file.filename):
            return templates.TemplateResponse(
                "add_face.html",
                {"request": request, "error": "Only image files are allowed"}
            )

        contents = await file.read()

        event_folder = os.path.join(UPLOAD_DIR, event_id)
        os.makedirs(event_folder, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(event_folder, unique_name)

        with open(file_path, "wb") as f:
            f.write(contents)

        image = decode_image(contents)

        face_data = extract_face_data(image)
        if not face_data:
            return templates.TemplateResponse(
                "add_face.html",
                {
                    "request": request,
                    "error": "No face detected"
                }
            )

        embeddings = [fd["embedding"] for fd in face_data]
        bboxes = [fd["bbox"] for fd in face_data]

        image_url = f"/uploads/{event_id}/{unique_name}"

        add_embeddings(
            embeddings=embeddings,
            image_url=image_url,
            event_id=event_id,
            bboxes=bboxes
        )

        # Gallery for event folder
        gallery = []
        for fname in sorted(os.listdir(event_folder)):
            if is_allowed_image(fname):
                gallery.append(f"/uploads/{event_id}/{fname}")

        return templates.TemplateResponse(
            "add_face.html",
            {
                "request": request,
                "success": True,
                "faces_added": len(embeddings),
                "event_id": event_id,
                "image_url": image_url,
                "gallery": gallery
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "add_face.html",
            {"request": request, "error": str(e)}
        )

        
# =====================================================
# SEARCH FACE (API JSON)
# =====================================================
@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = 50
):
    if not is_allowed_image(file.filename):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    contents = await file.read()
    image = decode_image(contents)

    face_data = extract_face_data(image)
    if not face_data:
        raise HTTPException(status_code=400, detail="No face detected")

    all_matches: List[Dict] = []

    for fd in face_data:
        matches = search_embedding(
            embedding=fd["embedding"],
            top_k=top_k,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        all_matches.extend(matches)

    # Deduplicate by metadata "id"
    unique = {}
    for m in all_matches:
        unique[m["id"]] = m

    results = list(unique.values())
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Group by event_id
    grouped: Dict[str, List[Dict]] = {}
    for r in results:
        grouped.setdefault(r["event_id"], []).append(r)

    return {
        "matches": grouped,
        "total_matches": len(results),
        "threshold": SIMILARITY_THRESHOLD
    }


# =====================================================
# SEARCH FACE (UI POST)
# =====================================================
@app.post("/ui/search")
async def search_face_ui_post(
    request: Request,
    file: UploadFile = File(...)
):
    if not is_allowed_image(file.filename):
        return templates.TemplateResponse(
            "search_face.html",
            {"request": request, "results": {}, "error": "Only image files are allowed"}
        )

    contents = await file.read()
    image = decode_image(contents)

    face_data = extract_face_data(image)
    if not face_data:
        return templates.TemplateResponse(
            "search_face.html",
            {
                "request": request,
                "results": {},
                "error": "No face detected"
            }
        )

    all_matches: List[Dict] = []
    for fd in face_data:
        all_matches.extend(
            search_embedding(
                embedding=fd["embedding"],
                top_k=50,
                similarity_threshold=SIMILARITY_THRESHOLD
            )
        )

    # Deduplicate by face-id
    unique = {}
    for m in all_matches:
        unique[m["id"]] = m

    results = list(unique.values())
    results.sort(key=lambda x: x["similarity"], reverse=True)

    grouped: Dict[str, List[Dict]] = {}
    for r in results:
        grouped.setdefault(r["event_id"], []).append(r)

    return templates.TemplateResponse(
        "search_face.html",
        {
            "request": request,
            "results": grouped,
            "threshold": SIMILARITY_THRESHOLD
        }
    )


# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        **stats()
    }
