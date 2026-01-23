import os
import uuid

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

from face_engine import decode_image, extract_face_embeddings
from faiss_index import (
    add_embeddings,
    search_embedding,
    stats,
    metadata
)

# =====================================================
# CONFIG
# =====================================================
UPLOAD_DIR = "uploads"
DISTANCE_THRESHOLD = 1.4

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
    contents = await file.read()

    event_folder = os.path.join(UPLOAD_DIR, event_id)
    os.makedirs(event_folder, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(event_folder, unique_name)

    with open(file_path, "wb") as f:
        f.write(contents)

    image = decode_image(contents)
    embeddings = extract_face_embeddings(image)

    if not embeddings:
        raise HTTPException(status_code=400, detail="No face detected")

    image_url = f"/uploads/{event_id}/{unique_name}"

    add_embeddings(
        embeddings=embeddings,
        image_url=image_url,
        event_id=event_id
    )

    return {
        "status": "success",
        "faces_added": len(embeddings),
        "image_url": image_url
    }

# =====================================================
# SEARCH FACE (API JSON)
# =====================================================
@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = 50
):
    contents = await file.read()
    image = decode_image(contents)

    embeddings = extract_face_embeddings(image)
    if not embeddings:
        raise HTTPException(status_code=400, detail="No face detected")

    all_matches = []
    for emb in embeddings:
        all_matches.extend(
            search_embedding(
                embedding=emb,
                top_k=top_k,
                distance_threshold=DISTANCE_THRESHOLD
            )
        )

    matched_events = set(m["event_id"] for m in all_matches)

    final_results = {}
    for m in metadata:
        if m["event_id"] in matched_events:
            final_results.setdefault(m["event_id"], []).append(m)

    return {"matches": final_results}

# =====================================================
# SEARCH FACE (UI POST)
# =====================================================
@app.post("/ui/search")
async def search_face_ui_post(
    request: Request,
    file: UploadFile = File(...)
):
    contents = await file.read()
    image = decode_image(contents)

    embeddings = extract_face_embeddings(image)
    if not embeddings:
        return templates.TemplateResponse(
            "search_face.html",
            {
                "request": request,
                "results": {},
                "error": "No face detected"
            }
        )

    all_matches = []
    for emb in embeddings:
        all_matches.extend(
            search_embedding(
                embedding=emb,
                top_k=50,
                distance_threshold=DISTANCE_THRESHOLD
            )
        )

    # 🔑 Build best-distance lookup per event
    distance_map = {}
    for m in all_matches:
        eid = m["event_id"]
        dist = m["distance"]
        if eid not in distance_map or dist < distance_map[eid]:
            distance_map[eid] = dist

    # 🔑 Merge distance into metadata (template-safe)
    final_results = {}
    for m in metadata:
        eid = m["event_id"]
        if eid in distance_map:
            enriched = dict(m)          # DO NOT mutate original metadata
            enriched["distance"] = distance_map[eid]
            final_results.setdefault(eid, []).append(enriched)

    return templates.TemplateResponse(
        "search_face.html",
        {
            "request": request,
            "results": final_results
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
