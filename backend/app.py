"""
backend/app.py  — FIXED VERSION
=================================
Key additions over original:
  1. /results/{video_stem}  now reads the _result.json that main.py writes
     (was returning empty / error before)
  2. /upload/ returns the full combined result (quality + shot_type + evaluation)
     immediately in the upload response — frontend doesn't need a second call
  3. CORS enabled so your frontend HTML (file://) can call the API
  4. /health now also reports whether the shot model is loaded
"""

import json
import os
import shutil
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.config as config
from main import analyze_video

app = FastAPI(title="Cricket Analytics API")

# ── CORS — lets any browser origin call the API (important for local dev) ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    ml_ok   = os.path.exists(os.path.join("models", "best_model.pkl"))
    shot_ok = os.path.exists(os.path.join("models", "shot_type_model.pth"))
    return {
        "status":       "ok",
        "ml_model":     "ready" if ml_ok   else "missing — run python train_model.py",
        "shot_model":   "ready" if shot_ok else "missing — run python train_shot_model.py",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upload + analyse video
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video file, runs the full analysis pipeline, and returns:
      - ML quality prediction  (label, confidence, probabilities)
      - Shot type prediction   (shot_type, confidence, all_probabilities)
      - Per-aspect evaluation  (Footwork, Head Position, …)
      - annotated video filename (stream it via GET /video/{filename})
    """
    suffix      = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path    = tempfile.mktemp(suffix=suffix)

    try:
        # Save upload to a temp file
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run analysis
        result = analyze_video(tmp_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Video processing failed")

        # Read evaluation.json written by save_evaluation()
        eval_path = os.path.join(OUTPUT_DIR, "evaluation.json")
        evaluation = {}
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                evaluation = json.load(f)

        annotated_name = result.get("annotated_video_filename") or os.path.basename(
            result.get("annotated_video", "")
        )

        return {
            "message":       "Processed",
            "filename":      annotated_name,
            "frame_count":   result.get("frame_count"),
            "avg_fps":       result.get("avg_fps"),
            "ml_prediction": result.get("quality"),      # may be None if model missing
            "shot_type":     result.get("shot_type"),    # {shot_type, confidence, all_probabilities}
            "evaluation":    evaluation,                  # {Footwork: {score, feedback}, …}
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Stream annotated video
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/video/{filename}")
def stream_video(filename: str):
    """Stream an annotated video file."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")

    def _iter(path, chunk=1 << 20):
        with open(path, "rb") as f:
            while chunk_data := f.read(chunk):
                yield chunk_data

    return StreamingResponse(
        _iter(path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX: Fetch saved results for a video
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/results/{video_stem}")
def get_results(video_stem: str):
    """
    Return the full result JSON for a previously analysed video.
    video_stem = filename without extension, e.g. '1.1'  (not '1.1.mp4')
    """
    result_path = os.path.join(OUTPUT_DIR, f"{video_stem}_result.json")
    eval_path   = os.path.join(OUTPUT_DIR, "evaluation.json")

    if not os.path.exists(result_path):
        raise HTTPException(
            status_code=404,
            detail=f"Results not found for '{video_stem}'. "
                   "Run main.py on this video first.",
        )

    with open(result_path) as f:
        result = json.load(f)

    # Also attach evaluation scores if present
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            result["evaluation"] = json.load(f)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model info endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    from src.ml_model import MLPredictor
    try:
        p = MLPredictor(model_path=os.path.join("models", "best_model.pkl"))
        return {"model_name": p._model_name, "features": p._feature_cols}
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/comparison")
def model_comparison():
    path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run train_model.py first")
    with open(path) as f:
        return json.load(f)