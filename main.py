# """
# FastAPI app for Brain Tumor MRI Classification using EffViT-Hybrid.

# Endpoints:
#     GET  /                  -> HTML upload page
#     POST /predict           -> JSON: {prediction, confidence, probabilities, gradcam_url, original_url}
#     GET  /static/...        -> serves CSS, JS, uploads, and Grad-CAM result images
# """
# import os
# import uuid
# import io
# from pathlib import Path

# from fastapi import FastAPI, File, UploadFile, Request, HTTPException
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from PIL import Image

# from inference import predict

# # ----------------------------------------------------------------------
# # App setup
# # ----------------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = str(BASE_DIR / "models" / "EffViT_Hybrid_final.keras")
# STATIC_DIR = BASE_DIR / "static"
# UPLOAD_DIR = STATIC_DIR / "uploads"
# RESULT_DIR = STATIC_DIR / "results"
# TEMPLATES_DIR = BASE_DIR / "templates"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# RESULT_DIR.mkdir(parents=True, exist_ok=True)

# app = FastAPI(
#     title="Brain Tumor MRI Classification",
#     description="EffViT-Hybrid (EfficientNetB0 + ViT-Tiny) for 4-class brain tumor classification",
#     version="1.0.0",
# )

# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# # ----------------------------------------------------------------------
# # Utilities
# # ----------------------------------------------------------------------
# ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# MAX_BYTES = 10 * 1024 * 1024     # 10 MB


# def _validate_upload(filename: str, content: bytes):
#     if not filename:
#         raise HTTPException(status_code=400, detail="No filename provided.")
#     ext = Path(filename).suffix.lower()
#     if ext not in ALLOWED_EXTS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
#         )
#     if len(content) > MAX_BYTES:
#         raise HTTPException(status_code=400, detail="File too large (max 10 MB).")
#     try:
#         Image.open(io.BytesIO(content)).verify()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


# # ----------------------------------------------------------------------
# # Startup
# # ----------------------------------------------------------------------
# @app.on_event("startup")
# def _warmup_model():
#     """Load the model on startup so the first request doesn't time out."""
#     if os.path.exists(MODEL_PATH):
#         from inference import load_model
#         try:
#             load_model(MODEL_PATH)
#             print("✓ Model warmed up on startup.")
#         except Exception as e:
#             print(f"⚠ Model warmup failed (will retry per-request): {e}")
#     else:
#         print(f"⚠ Model file not found at {MODEL_PATH}. Place EffViT_Hybrid_final.keras in models/.")


# # ----------------------------------------------------------------------
# # Routes
# # ----------------------------------------------------------------------
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     model_present = os.path.exists(MODEL_PATH)
#     return templates.TemplateResponse(
#         request=request,
#         name="index.html",
#         context={"model_present": model_present},
#     )


# @app.post("/predict")
# async def predict_endpoint(file: UploadFile = File(...)):
#     if not os.path.exists(MODEL_PATH):
#         raise HTTPException(
#             status_code=503,
#             detail=(
#                 "Model file not found. Download EffViT_Hybrid_final.keras from your "
#                 "Kaggle output and place it in the models/ folder."
#             ),
#         )

#     content = await file.read()
#     _validate_upload(file.filename, content)

#     # Save original
#     uid = uuid.uuid4().hex[:12]
#     ext = Path(file.filename).suffix.lower()
#     original_filename = f"orig_{uid}{ext}"
#     original_path = UPLOAD_DIR / original_filename
#     with open(original_path, "wb") as f:
#         f.write(content)

#     # Predict
#     try:
#         result = predict(content, MODEL_PATH)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

#     return JSONResponse({
#         "ok": True,
#         "predicted_class": result["predicted_class"],
#         "predicted_key": result["predicted_key"],
#         "confidence": result["confidence"],
#         "probabilities": result["probabilities"],
#         "original_url": f"/static/uploads/{original_filename}",
#         "gradcam_url": None,
#     })


# @app.get("/health")
# async def health():
#     return {
#         "status": "ok",
#         "model_present": os.path.exists(MODEL_PATH),
#         "model_path": MODEL_PATH,
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


"""
FastAPI app for Brain Tumor MRI Classification using EffViT-Hybrid.

Endpoints:
    GET  /                  -> HTML upload page
    POST /predict           -> JSON: {prediction, confidence, probabilities, gradcam_url, original_url}
    GET  /static/...        -> serves CSS, JS, uploads, and Grad-CAM result images
"""
import os
import uuid
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from inference import predict, compute_gradcam

# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "EffViT_Hybrid_final.keras")
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Brain Tumor MRI Classification",
    description="EffViT-Hybrid (EfficientNetB0 + ViT-Tiny) for 4-class brain tumor classification",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_BYTES = 10 * 1024 * 1024     # 10 MB


def _validate_upload(filename: str, content: bytes):
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")
    try:
        Image.open(io.BytesIO(content)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------
@app.on_event("startup")
def _warmup_model():
    """Load the model on startup so the first request doesn't time out."""
    if os.path.exists(MODEL_PATH):
        from inference import load_model
        try:
            load_model(MODEL_PATH)
            print("✓ Model warmed up on startup.")
        except Exception as e:
            print(f"⚠ Model warmup failed (will retry per-request): {e}")
    else:
        print(f"⚠ Model file not found at {MODEL_PATH}. Place EffViT_Hybrid_final.keras in models/.")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    model_present = os.path.exists(MODEL_PATH)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_present": model_present},
    )


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model file not found. Download EffViT_Hybrid_final.keras from your "
                "Kaggle output and place it in the models/ folder."
            ),
        )

    content = await file.read()
    _validate_upload(file.filename, content)

    # Save original
    uid = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix.lower()
    original_filename = f"orig_{uid}{ext}"
    original_path = UPLOAD_DIR / original_filename
    with open(original_path, "wb") as f:
        f.write(content)

    # Predict
    try:
        result = predict(content, MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Grad-CAM
    gradcam_url = None
    try:
        overlay, _, _ = compute_gradcam(content, MODEL_PATH, pred_index=result["predicted_idx"])
        if overlay is not None:
            gradcam_filename = f"gradcam_{uid}.jpg"
            gradcam_path = RESULT_DIR / gradcam_filename
            Image.fromarray(overlay).save(gradcam_path, quality=92)
            gradcam_url = f"/static/results/{gradcam_filename}"
    except Exception as e:
        print(f"⚠ Grad-CAM failed (non-fatal): {e}")

    return JSONResponse({
        "ok": True,
        "predicted_class": result["predicted_class"],
        "predicted_key": result["predicted_key"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "original_url": f"/static/uploads/{original_filename}",
        "gradcam_url": gradcam_url,
    })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_present": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
