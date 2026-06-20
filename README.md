# Brain Tumor MRI Classification — FastAPI Web Interface

A FastAPI web app that classifies brain MRI images into four categories — glioma, meningioma, pituitary tumor, or no tumor — using the EffViT-Hybrid model (EfficientNetB0 stem + ViT-Tiny encoder). Predictions are returned with class probabilities and a Grad-CAM attention overlay.

---

## Quick Start

### 1. Clone / extract this project

```bash
cd brain_tumor_app
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# OR
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the trained model

Download `EffViT_Hybrid_final.keras` from your Kaggle output and place it in:

```
brain_tumor_app/models/EffViT_Hybrid_final.keras
```

The app will detect it on startup. If missing, the UI shows a warning banner.

### 5. Run the server

```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

To allow access from another device on your network:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| GET    | `/`       | HTML upload page |
| POST   | `/predict`| Multipart upload (`file=<image>`), returns JSON |
| GET    | `/health` | Health check + model status |
| GET    | `/docs`   | Auto-generated OpenAPI docs |

### Example with `curl`

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@sample_mri.jpg"
```

### Response shape

```json
{
  "ok": true,
  "predicted_class": "Glioma",
  "predicted_key": "glioma",
  "confidence": 0.9412,
  "probabilities": [
    {"class": "glioma",     "name": "Glioma",          "prob": 0.9412},
    {"class": "meningioma", "name": "Meningioma",      "prob": 0.0421},
    {"class": "notumor",    "name": "No Tumor",        "prob": 0.0098},
    {"class": "pituitary",  "name": "Pituitary Tumor", "prob": 0.0069}
  ],
  "original_url": "/static/uploads/orig_abc123.jpg",
  "gradcam_url":  "/static/results/gradcam_abc123.jpg"
}
```

---

## Project Layout

```
brain_tumor_app/
├── main.py                # FastAPI application + routes
├── inference.py           # Model loading, prediction, Grad-CAM
├── custom_layers.py       # Custom Keras layers (TransformerBlock, AddPositionalEmbedding)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   ├── uploads/           # User-uploaded images (runtime)
│   └── results/           # Grad-CAM overlays (runtime)
└── models/
    └── EffViT_Hybrid_final.keras
```

---

## Notes

* **Model input.** EffViT-Hybrid expects raw pixel values in `[0, 255]` (not rescaled), because the underlying EfficientNetB0 stem has a built-in normalization layer.
* **Grad-CAM.** Computed on the `block5c_add` layer of the EfficientNetB0 stem, yielding a 14×14 attention map that is upsampled to 224×224 for overlay.
* **Hardware.** The model runs comfortably on CPU at ~150–500 ms per image. A GPU brings this down to ~10 ms.
* **Disclaimer.** This is a research prototype trained on the public Nickparvar Brain Tumor MRI Dataset. It is **not** a clinical diagnostic tool.

---

## License

For academic / educational use.
