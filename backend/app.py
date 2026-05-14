from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "data")).resolve()
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "data" / "models")).resolve()
UPLOAD_DIR = DATA_DIR / "input" / "uploads"
OUTPUT_DIR = DATA_DIR / "output"
STATIC_DIR = ROOT / "static"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_JSON_FIELDS = {
    "normalized_scores",
    "defects",
    "recommendations",
    "objects",
    "ocr",
    "faces",
    "ranking_factors",
}
NUMERIC_FIELDS = {
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
    "blur_score",
    "noise",
    "saturation",
    "resolution_mp",
    "dynamic_range",
    "colorfulness",
    "entropy",
    "shadow_clip_pct",
    "highlight_clip_pct",
    "edge_density",
    "texture_complexity",
    "quality_score",
    "overall_score",
    "text_count",
    "object_count",
    "face_count",
}
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
RUN_TTL_SECONDS = 24 * 60 * 60
MONGODB_URI_ENV = "MONGODB_URI"
MONGODB_DATABASE_ENV = "MONGODB_DATABASE"
MONGODB_COLLECTION_ENV = "MONGODB_COLLECTION"
DEFAULT_MONGODB_DATABASE = "image_quality_analysis"
DEFAULT_MONGODB_COLLECTION = "runs"
MONGODB_CLIENT: Any | None = None
MONGODB_COLLECTION: Any | None = None


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env(ROOT / ".env")

app = FastAPI(title="Multi-Image Quality Analysis and Ranking System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for folder in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "groq": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "ocr": "available" if ocr_available() else "optional",
        "yolo": "available" if yolo_available() else "optional",
        "mongodb": mongodb_status(),
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    cleanup_expired_runs()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        report = run_dir / "report.csv"
        if not report.exists():
            continue
        rows = read_report(report)
        ensure_xlsx(run_dir, rows)
        runs.append(
            {
                "id": run_dir.name,
                "created": run_dir.stat().st_mtime,
                "count": len(rows),
                "best": rows[0] if rows else None,
                "comparisonUrl": f"/outputs/{run_dir.name}/comparison.png"
                if (run_dir / "comparison.png").exists()
                else None,
                "reportUrl": f"/outputs/{run_dir.name}/report.xlsx",
                "csvUrl": f"/outputs/{run_dir.name}/report.csv",
            }
        )
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    cleanup_expired_runs()
    run_dir = safe_child(OUTPUT_DIR, run_id)
    report = run_dir / "report.csv"
    if not report.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    report_rows = read_report(report)
    ensure_xlsx(run_dir, report_rows)
    rows = add_urls(run_id, report_rows)
    return {
        "id": run_id,
        "rows": rows,
        "comparisonUrl": f"/outputs/{run_id}/comparison.png"
        if (run_dir / "comparison.png").exists()
        else None,
        "reportUrl": f"/outputs/{run_id}/report.xlsx",
        "csvUrl": f"/outputs/{run_id}/report.csv",
    }


@app.delete("/api/runs")
def clear_runs() -> dict[str, str]:
    clear_directory(UPLOAD_DIR)
    clear_directory(OUTPUT_DIR)
    return {"status": "cleared"}


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    cleanup_expired_runs()
    images = [file for file in files if Path(file.filename or "").suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one image file.")

    run_id = uuid.uuid4().hex
    upload_run_dir = UPLOAD_DIR / run_id
    output_run_dir = OUTPUT_DIR / run_id
    upload_run_dir.mkdir(parents=True, exist_ok=True)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for upload in images:
        filename = clean_filename(upload.filename or f"image-{len(rows) + 1}.jpg")
        unique_name = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}{Path(filename).suffix.lower()}"
        image_path = upload_run_dir / unique_name
        with image_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        row = score_image(image_path)
        row["original_filename"] = filename
        rows.append(row)
        annotated = output_run_dir / f"{Path(unique_name).stem}_analysis.png"
        write_analysis_image(image_path, annotated, row)

    rows.sort(key=lambda item: float(item["overall_score"]), reverse=True)
    report_path = output_run_dir / "report.csv"
    write_report(report_path, rows)
    write_xlsx(output_run_dir / "report.xlsx", rows)
    write_comparison(output_run_dir / "comparison.png", rows)
    save_run_to_mongodb(run_id, rows)

    return {
        "id": run_id,
        "rows": add_urls(run_id, rows),
        "comparisonUrl": f"/outputs/{run_id}/comparison.png",
        "reportUrl": f"/outputs/{run_id}/report.xlsx",
        "csvUrl": f"/outputs/{run_id}/report.csv",
    }


@app.post("/api/ask")
async def ask(payload: dict[str, Any]) -> dict[str, Any]:
    cleanup_expired_runs()
    question = str(payload.get("question", "")).strip()
    run_id = str(payload.get("runId", "")).strip()
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    if not question:
        return {
            "answer": "Ask about a completed analysis run or upload images first.",
            "sections": [],
            "source": "local",
        }

    rows: list[dict[str, Any]] = []
    if run_id:
        report = safe_child(OUTPUT_DIR, run_id) / "report.csv"
        if report.exists():
            rows = read_report(report)

    if not rows:
        runs = list_runs()
        if runs:
            latest = runs[0]["id"]
            report = safe_child(OUTPUT_DIR, latest) / "report.csv"
            rows = read_report(report)

    return answer_question(question, label_rows(rows), history)


def safe_child(parent: Path, child: str) -> Path:
    path = (parent / child).resolve()
    if parent.resolve() not in path.parents and path != parent.resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    return path


def clean_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip()
    return safe or "image.jpg"


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except PermissionError:
            continue


def cleanup_expired_runs(max_age_seconds: int = RUN_TTL_SECONDS) -> None:
    cutoff = time.time() - max_age_seconds
    run_ids = {
        child.name
        for root in (UPLOAD_DIR, OUTPUT_DIR)
        if root.exists()
        for child in root.iterdir()
        if child.is_dir()
    }
    for run_id in run_ids:
        paths = [UPLOAD_DIR / run_id, OUTPUT_DIR / run_id]
        mtimes = [path.stat().st_mtime for path in paths if path.exists()]
        if mtimes and max(mtimes) < cutoff:
            for path in paths:
                if path.exists():
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                    except PermissionError:
                        continue


def mongodb_configured() -> bool:
    return bool(os.getenv(MONGODB_URI_ENV))


def mongodb_status() -> str:
    if not mongodb_configured():
        return "optional"
    collection = mongodb_collection()
    if collection is None:
        return "unavailable"
    return "connected"


def mongodb_collection() -> Any | None:
    global MONGODB_CLIENT, MONGODB_COLLECTION
    if MONGODB_COLLECTION is not None:
        return MONGODB_COLLECTION
    uri = os.getenv(MONGODB_URI_ENV)
    if not uri:
        return None
    try:
        from pymongo import MongoClient

        database_name = os.getenv(MONGODB_DATABASE_ENV, DEFAULT_MONGODB_DATABASE)
        collection_name = os.getenv(MONGODB_COLLECTION_ENV, DEFAULT_MONGODB_COLLECTION)
        MONGODB_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=5000)
        MONGODB_CLIENT.admin.command("ping")
        MONGODB_COLLECTION = MONGODB_CLIENT[database_name][collection_name]
        return MONGODB_COLLECTION
    except Exception as exc:
        print(f"[MongoDB] Connection unavailable: {brief_error(exc)}")
        return None


def mongodb_run_document(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    local_created_at = created_at.astimezone()
    images = []
    for index, row in enumerate(rows, start=1):
        image = normalize_for_mongodb(row)
        image["rank"] = index
        image["original_filename"] = image.get("original_filename") or image.get("filename") or ""
        images.append(image)
    return {
        "run_id": run_id,
        "created_at": created_at,
        "created_at_unix": created_at.timestamp(),
        "entered_at_utc": created_at.isoformat(),
        "entered_at_local": local_created_at.isoformat(),
        "entered_date_local": local_created_at.strftime("%Y-%m-%d"),
        "entered_time_local": local_created_at.strftime("%H:%M:%S %Z"),
        "image_count": len(images),
        "best_overall_score": float(images[0].get("overall_score") or 0) if images else 0,
        "images": images,
    }


def save_run_to_mongodb(run_id: str, rows: list[dict[str, Any]]) -> bool:
    collection = mongodb_collection()
    if collection is None:
        print(f"[MongoDB] Skipped save for run {run_id}: MongoDB is not configured or connection is unavailable.")
        return False
    document = mongodb_run_document(run_id, rows)
    try:
        result = collection.replace_one({"run_id": run_id}, document, upsert=True)
        print(
            f"[MongoDB] Saved run {run_id} to "
            f"{os.getenv(MONGODB_DATABASE_ENV, DEFAULT_MONGODB_DATABASE)}."
            f"{os.getenv(MONGODB_COLLECTION_ENV, DEFAULT_MONGODB_COLLECTION)} "
            f"with {len(rows)} image(s); matched={result.matched_count}, upserted={result.upserted_id is not None}."
        )
        return True
    except Exception as exc:
        print(f"[MongoDB] Save failed: {brief_error(exc)}")
        return False


def normalize_for_mongodb(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_for_mongodb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_for_mongodb(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_mongodb(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def score_image(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not read {path.name}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray.astype("float32") - denoised.astype("float32")))
    saturation = float(np.mean(hsv[:, :, 1]))
    resolution_mp = width * height / 1_000_000
    dynamic_range = float(np.max(gray) - np.min(gray))
    entropy = image_entropy(gray)
    shadow_clip_pct = float(np.mean(gray <= 5) * 100)
    highlight_clip_pct = float(np.mean(gray >= 250) * 100)
    colorfulness = image_colorfulness(image)
    blur_score = blur_risk_from_sharpness(sharpness)
    edge_density = float(np.mean(cv2.Canny(gray, 80, 180) > 0) * 100)
    texture_complexity = float(np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))

    channels = cv2.mean(image)[:3]
    color_cast = classify_color_cast(channels)
    exposure = classify_exposure(brightness)
    faces = detect_faces(image)
    objects = detect_objects(path)
    ocr = detect_text(path)

    normalized_scores = {
        "brightness_balance": clamp(100 - abs(brightness - 127) / 127 * 100),
        "contrast": clamp(contrast / 75 * 100),
        "sharpness": clamp(sharpness / 350 * 100),
        "noise_control": clamp(100 - noise / 25 * 100),
        "saturation": clamp(saturation / 150 * 100),
        "dynamic_range": clamp(dynamic_range / 255 * 100),
        "clipping_control": clamp(100 - (shadow_clip_pct + highlight_clip_pct) * 5),
        "resolution": clamp(resolution_mp / 4 * 100),
        "edge_detail": clamp(edge_density / 12 * 100),
        "texture_detail": clamp(texture_complexity / 55 * 100),
    }
    quality_keys = (
        "brightness_balance",
        "contrast",
        "sharpness",
        "noise_control",
        "saturation",
        "dynamic_range",
        "clipping_control",
        "resolution",
    )
    quality_score = sum(normalized_scores[key] for key in quality_keys) / len(quality_keys)
    overall_score = quality_score
    defects = detect_defects(
        brightness,
        contrast,
        sharpness,
        noise,
        saturation,
        shadow_clip_pct,
        highlight_clip_pct,
        color_cast,
        resolution_mp,
    )
    recommendations = build_recommendations(defects, ocr, objects, faces)
    ranking_factors = build_ranking_factors(normalized_scores)

    return {
        "filepath": str(path),
        "filename": path.name,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "blur_score": round(blur_score, 3),
        "noise": round(noise, 2),
        "saturation": round(saturation, 2),
        "exposure": exposure,
        "color_cast": color_cast,
        "resolution_mp": round(resolution_mp, 2),
        "dynamic_range": round(dynamic_range, 2),
        "colorfulness": round(colorfulness, 2),
        "entropy": round(entropy, 2),
        "shadow_clip_pct": round(shadow_clip_pct, 2),
        "highlight_clip_pct": round(highlight_clip_pct, 2),
        "edge_density": round(edge_density, 2),
        "texture_complexity": round(texture_complexity, 2),
        "quality_score": round(quality_score, 1),
        "overall_score": round(overall_score, 1),
        "text_count": int(ocr["count"]),
        "object_count": len(objects["items"]),
        "face_count": int(faces["count"]),
        "ocr": ocr,
        "objects": objects,
        "faces": faces,
        "defects": defects,
        "recommendations": recommendations,
        "ranking_factors": ranking_factors,
        "normalized_scores": {key: round(value, 1) for key, value in normalized_scores.items()},
    }


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def image_colorfulness(image: np.ndarray) -> float:
    b, g, r = cv2.split(image.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def detect_text(path: Path) -> dict[str, Any]:
    try:
        import pytesseract

        configure_tesseract(pytesseract)
        image = cv2.imread(str(path))
        if image is None:
            return {"available": True, "count": 0, "items": [], "summary": "No readable text detected."}
        raw = best_ocr_result(pytesseract, image)
        scale = float(raw.pop("_scale", 1) or 1)
    except Exception as exc:
        return {"available": False, "count": 0, "items": [], "summary": f"OCR unavailable: {brief_error(exc)}"}

    items = []
    for index, text in enumerate(raw.get("text", [])):
        value = clean_ocr_token(str(text).strip())
        confidence = float(to_number(raw.get("conf", ["0"])[index]))
        if not is_reliable_ocr_token(value, confidence):
            continue
        items.append(
            {
                "text": value,
                "confidence": round(confidence, 1),
                "box": [
                    int(int(raw.get("left", [0])[index]) / scale),
                    int(int(raw.get("top", [0])[index]) / scale),
                    int(int(raw.get("width", [0])[index]) / scale),
                    int(int(raw.get("height", [0])[index]) / scale),
                ],
            }
        )
    illustration_like = is_likely_illustration(image)
    if not has_reliable_text_evidence(items, illustration_like):
        return {
            "available": True,
            "count": 0,
            "items": [],
            "summary": "No reliable readable text detected.",
            "candidates": len(items),
        }
    transcript = " ".join(item["text"] for item in items[:40])
    return {
        "available": True,
        "count": len(items),
        "items": items[:24],
        "summary": transcript[:240] if transcript else "No readable text detected.",
    }


def best_ocr_result(pytesseract_module: Any, image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = 2 if max(width, height) < 1800 else 1
    if scale > 1:
        gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    contrast = cv2.convertScaleAbs(gray, alpha=1.35, beta=8)
    threshold = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    variants = (gray, contrast, threshold)
    configs = ("--psm 6", "--psm 11")
    best: dict[str, Any] | None = None
    best_score = -1
    for variant in variants:
        for config in configs:
            raw = pytesseract_module.image_to_data(variant, output_type=pytesseract_module.Output.DICT, config=config)
            raw["_scale"] = scale
            tokens = [
                str(text).strip()
                for index, text in enumerate(raw.get("text", []))
                if is_reliable_ocr_token(clean_ocr_token(str(text).strip()), float(to_number(raw.get("conf", ["0"])[index])))
            ]
            score = sum(len(token) for token in tokens) + len(tokens) * 4
            if score > best_score:
                best = raw
                best_score = score
    return best or {"text": []}


def clean_ocr_token(value: str) -> str:
    return "".join(char for char in value.strip() if char.isalnum() or char in "-./%")


def is_reliable_ocr_token(value: str, confidence: float) -> bool:
    alnum = [char for char in value if char.isalnum()]
    if confidence < 55 or len(alnum) < 3:
        return False
    letters = [char for char in alnum if char.isalpha()]
    digits = [char for char in alnum if char.isdigit()]
    if digits and len(alnum) >= 3:
        return True
    if len(letters) < 3:
        return False
    vowel_count = sum(1 for char in letters if char.lower() in "aeiou")
    if vowel_count > 0:
        return True
    return confidence >= 82 and len(letters) >= 4


def has_reliable_text_evidence(items: list[dict[str, Any]], illustration_like: bool = False) -> bool:
    if not items:
        return False
    total_chars = sum(len("".join(char for char in str(item.get("text", "")) if char.isalnum())) for item in items)
    strong_items = [item for item in items if float(item.get("confidence") or 0) >= 75]
    if illustration_like:
        return total_chars >= 18 and len(strong_items) >= 4
    return total_chars >= 8 and (len(items) >= 2 or bool(strong_items))


def detect_objects(path: Path) -> dict[str, Any]:
    model_path = find_yolo_model()
    if model_path is None:
        return {"available": False, "model": None, "items": [], "summary": "YOLO model not configured."}
    try:
        configure_ultralytics()
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        result = model(str(path), verbose=False)[0]
    except Exception as exc:
        return {"available": False, "model": str(model_path.name), "items": [], "summary": f"YOLO unavailable: {brief_error(exc)}"}

    items = []
    names = result.names or {}
    for box in result.boxes:
        cls = int(box.cls[0])
        xyxy = [round(float(value), 1) for value in box.xyxy[0].tolist()]
        items.append(
            {
                "label": str(names.get(cls, cls)),
                "confidence": round(float(box.conf[0]), 3),
                "box": xyxy,
            }
        )
    items.sort(key=lambda item: item["confidence"], reverse=True)
    labels = ", ".join(sorted({item["label"] for item in items[:8]}))
    return {
        "available": True,
        "model": str(model_path.name),
        "items": items[:30],
        "summary": labels or "No objects detected.",
    }


def detect_faces(image: np.ndarray) -> dict[str, Any]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return {"available": False, "count": 0, "items": [], "summary": "OpenCV face cascade unavailable."}
    if is_likely_illustration(image):
        return {"available": True, "count": 0, "items": [], "summary": "No real-photo faces detected."}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(str(cascade_path))
    height, width = gray.shape[:2]
    min_face_size = int(max(96, min(width, height) * 0.12))
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=9,
        minSize=(min_face_size, min_face_size),
    )
    items = []
    for x, y, w, h in faces:
        aspect = w / max(h, 1)
        if 0.78 <= aspect <= 1.25 and is_likely_real_face(image[y : y + h, x : x + w]):
            items.append({"box": [int(x), int(y), int(w), int(h)]})
    return {
        "available": True,
        "count": len(items),
        "items": items[:20],
        "summary": f"{len(items)} face{'s' if len(items) != 1 else ''} detected.",
    }


def is_likely_real_face(face_roi: np.ndarray) -> bool:
    if face_roi.size == 0:
        return False
    height, width = face_roi.shape[:2]
    if min(height, width) < 96:
        return False
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    skin_hsv = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 180) & (hsv[:, :, 2] >= 50)
    skin_ycrcb = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
    skin_ratio = float(np.mean(skin_hsv & skin_ycrcb))
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    return skin_ratio >= 0.08 and texture >= 18


def is_likely_illustration(image: np.ndarray) -> bool:
    if image.size == 0:
        return False
    sample = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    smooth_ratio = float(np.mean(gradient < 8))
    saturation = float(np.mean(hsv[:, :, 1]))
    palette_size = len(np.unique((sample // 16).reshape(-1, 3), axis=0))
    return smooth_ratio > 0.5 and saturation > 100 and palette_size < 380


def find_yolo_model() -> Path | None:
    candidates = [
        MODEL_DIR / "yolov8n.pt",
        MODEL_DIR / "yolov8s.pt",
        MODEL_DIR / "yolo11n.pt",
        DATA_DIR / "models" / "yolov8n.pt",
        DATA_DIR / "models" / "yolov8s.pt",
        DATA_DIR / "models" / "yolo11n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ocr_available() -> bool:
    try:
        import pytesseract

        configure_tesseract(pytesseract)
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def configure_tesseract(pytesseract_module: Any) -> None:
    configured = os.getenv("TESSERACT_CMD")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def yolo_available() -> bool:
    return find_yolo_model() is not None


def configure_ultralytics() -> None:
    config_dir = DATA_DIR / "tmp" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_SETTINGS", str(config_dir / "settings.json"))


def detect_defects(
    brightness: float,
    contrast: float,
    sharpness: float,
    noise: float,
    saturation: float,
    shadow_clip_pct: float,
    highlight_clip_pct: float,
    color_cast: str,
    resolution_mp: float,
) -> list[dict[str, str]]:
    defects: list[dict[str, str]] = []
    if brightness < 65:
        defects.append({"name": "Underexposed", "severity": "high", "detail": "The image is too dark for reliable inspection."})
    elif brightness > 190:
        defects.append({"name": "Overexposed", "severity": "high", "detail": "Highlights are too bright and may lose detail."})
    if contrast < 35:
        defects.append({"name": "Low contrast", "severity": "medium", "detail": "Tone separation is weak."})
    if sharpness < 60:
        defects.append({"name": "Out of focus", "severity": "high", "detail": "Edges are very soft, suggesting focus or motion blur."})
    elif sharpness < 110:
        defects.append({"name": "Soft focus", "severity": "medium", "detail": "Fine text and edges are softer than ideal."})
    elif sharpness < 150:
        defects.append({"name": "Slightly soft", "severity": "low", "detail": "The image is usable, but small label text may benefit from sharper focus."})
    if noise > 20:
        defects.append({"name": "Visible noise", "severity": "medium", "detail": "Grain or sensor noise is affecting clean detail."})
    if saturation < 35:
        defects.append({"name": "Muted color", "severity": "low", "detail": "Color intensity is low."})
    elif saturation > 175:
        defects.append({"name": "Oversaturation", "severity": "medium", "detail": "Colors may look unnatural."})
    if shadow_clip_pct > 2:
        defects.append({"name": "Shadow clipping", "severity": "medium", "detail": "Dark areas contain blocked blacks."})
    if highlight_clip_pct > 2:
        defects.append({"name": "Highlight clipping", "severity": "medium", "detail": "Bright areas contain clipped whites."})
    if color_cast != "neutral":
        defects.append({"name": f"{color_cast.title()} color cast", "severity": "low", "detail": "White balance appears shifted."})
    if resolution_mp < 0.7:
        defects.append({"name": "Low resolution", "severity": "medium", "detail": "The image may not hold enough detail for cropping or OCR."})
    return defects


def build_recommendations(defects: list[dict[str, str]], ocr: dict[str, Any], objects: dict[str, Any], faces: dict[str, Any]) -> list[str]:
    names = {defect["name"].lower() for defect in defects}
    recommendations = []
    if "underexposed" in names:
        recommendations.append("Increase exposure or add front lighting before capture.")
    if "overexposed" in names or "highlight clipping" in names:
        recommendations.append("Reduce exposure and protect highlights during capture.")
    if "out of focus" in names or "soft focus" in names:
        recommendations.append("Use a faster shutter speed, tripod, or refocus on the main subject.")
    if "slightly soft" in names and ocr.get("available") and int(ocr.get("count") or 0) > 0:
        recommendations.append("Crop closer or tap-focus on the label if exact text reading matters.")
    if "visible noise" in names:
        recommendations.append("Lower ISO or improve lighting to reduce grain.")
    if "low contrast" in names:
        recommendations.append("Add local contrast or reshoot with more directional light.")
    if any("color cast" in name for name in names):
        recommendations.append("Correct white balance using a neutral reference.")
    if objects.get("available") and not objects.get("items"):
        recommendations.append("Make the main subject larger and reduce background clutter if object detection matters.")
    if faces.get("available") and int(faces.get("count") or 0) > 0:
        recommendations.append("Keep faces sharp and evenly lit if the image is intended for people detection.")
    if not recommendations:
        recommendations.append("Image quality is balanced; minor edits should be conservative.")
    return recommendations[:7]


def build_ranking_factors(normalized_scores: dict[str, float]) -> list[str]:
    ordered = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
    strengths = ", ".join(name.replace("_", " ") for name, _ in ordered[:3])
    weaknesses = ", ".join(name.replace("_", " ") for name, _ in ordered[-2:])
    return [
        f"Strongest quality factors: {strengths}.",
        f"Weakest quality factors: {weaknesses}.",
        "Ranking is based on image quality metrics, not detected faces, text, or objects.",
    ]


def blur_risk_from_sharpness(sharpness: float) -> float:
    return clamp(1 - sharpness / 260, 0, 1)


def brief_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:140] or exc.__class__.__name__


def classify_exposure(brightness: float) -> str:
    if brightness < 65:
        return "underexposed"
    if brightness > 190:
        return "overexposed"
    return "normal"


def classify_color_cast(channels: tuple[float, float, float]) -> str:
    blue, green, red = channels
    if red - blue > 18:
        return "warm"
    if blue - red > 18:
        return "cool"
    if green - ((red + blue) / 2) > 18:
        return "green"
    return "neutral"


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "filepath",
        "filename",
        "original_filename",
        "width",
        "height",
        "brightness",
        "contrast",
        "sharpness",
        "blur_score",
        "noise",
        "saturation",
        "exposure",
        "color_cast",
        "resolution_mp",
        "dynamic_range",
        "colorfulness",
        "entropy",
        "shadow_clip_pct",
        "highlight_clip_pct",
        "edge_density",
        "texture_complexity",
        "quality_score",
        "overall_score",
        "text_count",
        "object_count",
        "face_count",
        "defects",
        "recommendations",
        "ocr",
        "objects",
        "faces",
        "ranking_factors",
        "normalized_scores",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for field in REPORT_JSON_FIELDS:
                item[field] = json.dumps(item.get(field, [] if field != "normalized_scores" else {}))
            writer.writerow(item)


def read_report(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("feature_score", None)
        for key in NUMERIC_FIELDS:
            row[key] = to_number(row.get(key))
        for key in REPORT_JSON_FIELDS:
            fallback: Any = {} if key in {"normalized_scores", "ocr", "objects", "faces"} else []
            row[key] = parse_json_field(row.get(key), fallback)
        row.setdefault("original_filename", row.get("filename", ""))
        row.setdefault("quality_score", row.get("overall_score", 0))
        row["overall_score"] = row.get("quality_score") or row.get("overall_score") or 0
        if not row.get("defects"):
            row["defects"] = detect_defects(
                float(row.get("brightness") or 0),
                float(row.get("contrast") or 0),
                float(row.get("sharpness") or 0),
                float(row.get("noise") or 0),
                float(row.get("saturation") or 0),
                float(row.get("shadow_clip_pct") or 0),
                float(row.get("highlight_clip_pct") or 0),
                str(row.get("color_cast") or "neutral"),
                float(row.get("resolution_mp") or 0),
            )
        if not row.get("recommendations"):
            row["recommendations"] = build_recommendations(
                row.get("defects") or [],
                row.get("ocr") or {},
                row.get("objects") or {},
                row.get("faces") or {},
            )
        row.setdefault("ranking_factors", [])
    rows.sort(key=lambda item: float(item["overall_score"]), reverse=True)
    return rows


def parse_json_field(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        try:
            return json.loads(str(raw).replace("'", '"'))
        except json.JSONDecodeError:
            return fallback


def add_urls(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir = OUTPUT_DIR / run_id
    output = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["image_index"] = index
        item["display_label"] = f"Image {index}"
        item["original_filename"] = item.get("original_filename") or item.get("filename", "")
        source_path = Path(str(item.get("filepath", "")))
        if source_path.exists():
            item["sourceUrl"] = f"/uploads/{run_id}/{source_path.name}"
        analysis = output_dir / f"{source_path.stem}_analysis.png"
        if analysis.exists():
            item["analysisUrl"] = f"/outputs/{run_id}/{analysis.name}"
        output.append(item)
    return output


def label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["image_index"] = index
        item["display_label"] = f"Image {index}"
        item.setdefault("original_filename", item.get("filename", ""))
        labeled.append(item)
    return labeled


def row_label(row: dict[str, Any]) -> str:
    return str(row.get("display_label") or f"Image {row.get('image_index') or '?'}")


def to_number(value: Any) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number.is_integer():
        return int(number)
    return number


METRIC_ALIASES = {
    "brightness": ("brightness", "Brightness"),
    "contrast": ("contrast", "Contrast"),
    "sharpness": ("sharpness", "Sharpness"),
    "sharp": ("sharpness", "Sharpness"),
    "blur": ("blur_score", "Blur risk"),
    "noise": ("noise", "Noise"),
    "saturation": ("saturation", "Saturation"),
    "resolution": ("resolution_mp", "Resolution"),
    "dynamic range": ("dynamic_range", "Dynamic range"),
    "colorfulness": ("colorfulness", "Color richness"),
    "color richness": ("colorfulness", "Color richness"),
    "color cast": ("color_cast", "Color cast"),
    "shadow clipping": ("shadow_clip_pct", "Shadow clipping"),
    "highlight clipping": ("highlight_clip_pct", "Highlight clipping"),
    "edge": ("edge_density", "Detail edges"),
    "texture": ("texture_complexity", "Texture detail"),
    "entropy": ("entropy", "Tonal entropy"),
    "text": ("text_count", "Text detected"),
    "ocr": ("text_count", "Text detected"),
    "face": ("face_count", "Faces detected"),
    "object": ("object_count", "Objects detected"),
    "score": ("overall_score", "Overall score"),
    "quality": ("quality_score", "Quality score"),
    "exposure": ("exposure", "Exposure"),
}


def requested_image_index(question: str) -> int | None:
    match = re.search(r"\b(?:image|img|imge)\s*#?\s*(\d+)\b", question.lower())
    if not match:
        return None
    return int(match.group(1))


def requested_metric(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    for phrase, metric in sorted(METRIC_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in lower:
            return metric
    return None


def metric_value_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if key == "resolution_mp":
        return f"{value} MP"
    if key in {"shadow_clip_pct", "highlight_clip_pct", "edge_density"}:
        return f"{value}%"
    return str(value)


def weak_metric_items(row: dict[str, Any]) -> list[str]:
    scores = row.get("normalized_scores") or {}
    ordered = sorted(scores.items(), key=lambda item: float(item[1]))
    items = [f"{name.replace('_', ' ').title()}: {value}/100" for name, value in ordered[:4]]
    defects = row.get("defects") or []
    items.extend(f"{defect.get('name')}: {defect.get('detail', '')}".strip() for defect in defects[:4])
    return items or ["No major weak metric was recorded."]


def answer_question(question: str, rows: list[dict[str, Any]], history: list[Any] | None = None) -> dict[str, Any]:
    groq = ask_groq(question, rows, history or [])
    if groq:
        return {**groq, "source": "groq"}

    if not rows:
        return {
            "answer": (
                "I can chat normally, but the LLM service is not responding right now. "
                "Once it is reachable, I can also use uploaded image-quality results as context."
            ),
            "sections": [],
            "source": "local",
        }

    lower = question.lower()
    best = rows[0]
    worst = rows[-1]
    image_index = requested_image_index(question)
    target_row = rows[image_index - 1] if image_index and 1 <= image_index <= len(rows) else None
    metric = requested_metric(question)

    if image_index and target_row is None:
        return structured_answer(f"This run has only {len(rows)} image(s), so I cannot find Image {image_index}.")

    if target_row and metric:
        key, label = metric
        return structured_answer(f"{label} for {row_label(target_row)} is {metric_value_text(target_row, key)}.")

    if target_row and any(word in lower for word in ("why", "rank", "last", "lowest", "bad", "worse", "worst")):
        rank = int(target_row.get("image_index") or image_index or 0)
        if target_row == worst:
            answer = f"{row_label(target_row)} ranks last because it has the lowest overall score in this run: {target_row.get('overall_score')}."
        else:
            answer = f"{row_label(target_row)} is ranked #{rank} of {len(rows)} with score {target_row.get('overall_score')}; it is not the last image in this run."
        return structured_answer(
            answer,
            [
                {"title": "Weakest factors", "items": weak_metric_items(target_row)},
                {"title": "Recommended fixes", "items": target_row.get("recommendations") or ["No specific fix recorded."]},
            ],
        )

    if any(phrase in lower for phrase in ("who are you", "what is this", "what are you", "hello", "hi", "hey")):
        return structured_answer(
            "I am the image quality assistant for this website. I use the metrics engine output for the uploaded images, then explain ranking, defects, and fixes in plain language.",
            [
                {
                    "title": "What I can answer",
                    "items": [
                        f"Which image is best: currently {row_label(best)} with score {best.get('overall_score')}.",
                        "Why an image ranked higher or lower.",
                        "Defects like underexposure, blur, noise, low contrast, clipping, and color cast.",
                        "Fix recommendations based on those defects.",
                    ],
                }
            ],
        )
    if any(word in lower for word in ("defect", "problem", "issue", "fix", "recommend", "improve", "compare")):
        sections = []
        for row in rows[:8]:
            defects = row.get("defects") or []
            recs = row.get("recommendations") or []
            items = []
            if defects:
                items.extend(f"{defect.get('name')}: {defect.get('detail', '')}".strip() for defect in defects[:4])
            else:
                items.append("No major defects detected.")
            items.extend(f"Fix: {rec}" for rec in recs[:3])
            sections.append({"title": f"{row_label(row)} findings", "items": items})
        return structured_answer(
            f"I compared {len(rows)} ranked images and listed the defects plus fixes by image label.",
            sections,
        )
    if any(word in lower for word in ("best", "top", "winner", "highest")):
        return structured_answer(
            f"The best ranked image is {row_label(best)} with an overall score of {best['overall_score']}.",
            [
                {"title": "Why it ranks first", "items": best.get("ranking_factors") or []},
                {"title": "Recommended fixes", "items": best.get("recommendations") or []},
            ],
        )
    if any(word in lower for word in ("worst", "lowest", "bad", "last")):
        return structured_answer(
            f"The lowest ranked image is {row_label(worst)} with an overall score of {worst['overall_score']}.",
            [
                {"title": "Detected defects", "items": [defect["name"] + ": " + defect["detail"] for defect in worst.get("defects", [])]},
                {"title": "Recommended fixes", "items": worst.get("recommendations") or []},
            ],
        )
    if "sharp" in lower or "blur" in lower:
        sharpest = max(rows, key=lambda item: float(item.get("sharpness") or 0))
        return structured_answer(
            f"The sharpest image is {row_label(sharpest)} with a sharpness value of {sharpest['sharpness']}.",
            [{"title": "Ranking context", "items": sharpest.get("ranking_factors") or []}],
        )
    if "bright" in lower or "exposure" in lower or "dark" in lower:
        normal = [row for row in rows if row.get("exposure") == "normal"]
        if normal:
            names = ", ".join(row_label(row) for row in normal[:3])
            return structured_answer(f"Images with normal exposure include: {names}.")
        return structured_answer(f"The run has no image marked as normal exposure. The best overall option is {row_label(best)}.")
    if "text" in lower or "ocr" in lower:
        readable = [row for row in rows if int(row.get("text_count") or 0) > 0]
        items = [f"{row_label(row)}: {row.get('ocr', {}).get('summary', '')}" for row in readable[:5]]
        return structured_answer(
            f"{len(readable)} image{'s' if len(readable) != 1 else ''} have text detected.",
            [{"title": "Text detected", "items": items or ["No readable text detected in this run."]}],
        )
    if "face" in lower:
        face_rows = [row for row in rows if int(row.get("face_count") or 0) > 0]
        items = [f"{row_label(row)}: {row['face_count']} face(s)" for row in face_rows[:8]]
        return structured_answer(
            f"{len(face_rows)} image{'s' if len(face_rows) != 1 else ''} contain detected faces.",
            [{"title": "Face findings", "items": items or ["No faces detected."]}],
        )
    if "object" in lower or "yolo" in lower:
        items = [f"{row_label(row)}: {row.get('objects', {}).get('summary', 'No objects')}" for row in rows[:8]]
        return structured_answer("Object detection findings are summarized below.", [{"title": "Objects detected", "items": items}])
    if "report" in lower or "excel" in lower or "xlsx" in lower:
        return structured_answer("Use the Excel Report button above the results. It downloads report.xlsx, which opens directly in Excel.")

    average = sum(float(row.get("overall_score") or 0) for row in rows) / len(rows)
    return structured_answer(
        f"This run contains {len(rows)} images. Best: {row_label(best)} ({best['overall_score']}). "
        f"Lowest: {row_label(worst)} ({worst['overall_score']}). Average score: {average:.1f}.",
        [
            {"title": "Best fixes to consider", "items": best.get("recommendations") or []},
            {"title": "Worst image defects", "items": [defect["name"] for defect in worst.get("defects", [])] or ["No major defects recorded."]},
        ],
    )


def structured_answer(answer: str, sections: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"answer": answer, "sections": sections or [], "source": "local"}


def ask_groq(question: str, rows: list[dict[str, Any]], history: list[Any]) -> dict[str, Any] | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful conversational assistant built into a multi-image quality analysis website. "
                "Answer the user's actual message naturally, including normal greetings and general questions. "
                "For casual greetings or small talk, respond briefly and naturally without dumping image metrics. "
                "If image-analysis data is supplied, use it as trusted context only when the user asks about rankings, scores, "
                "defects, text detected, objects detected, faces detected, reports, and improvement advice. "
                "If no image-analysis data is supplied, do not pretend images were analyzed; invite the user to upload images "
                "only when that is relevant. "
                "When discussing analyzed images, refer to them as Image 1, Image 2, etc. and avoid raw filenames unless asked. "
                "When the user asks for fixes, compare defects and give concrete correction recommendations per image. "
                "You must respond in valid JSON ONLY, with no markdown formatting, no code blocks, and no preamble. "
                "Return strict JSON with keys: 'answer' (string) and 'sections' (array of objects with 'title' (string) and 'items' (array of strings)). "
                "Use sections only when they make the answer clearer; for casual chat, return an empty sections array. "
                "Keep the tone helpful, direct, and chat-like, not robotic."
            ),
        }
    ]
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "has_image_analysis": bool(rows),
                    "run_summary": summarize_rows_for_ai(rows),
                },
                ensure_ascii=True,
            ),
        }
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=18) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        print(f"[Groq API] HTTP Error {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        return None
    except Exception as exc:
        print(f"[Groq API] Connection Error: {exc}")
        return None
    try:
        content = data["choices"][0]["message"]["content"].strip()
        
        # Clean up markdown blocks if the LLM ignored instructions
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        parsed = json.loads(content)
    except Exception as exc:
        print(f"[Groq API] Parse Error: {exc}\nRaw Content: {data.get('choices', [{}])[0].get('message', {}).get('content')}")
        return None
    answer = str(parsed.get("answer") or "").strip()
    sections = parsed.get("sections")
    if not answer:
        return None
    if not isinstance(sections, list):
        sections = []
    clean_sections = []
    for section in sections[:5]:
        if not isinstance(section, dict):
            continue
        items = section.get("items") if isinstance(section.get("items"), list) else []
        clean_sections.append({"title": str(section.get("title") or "Details"), "items": [str(item) for item in items[:6]]})
    return {"answer": answer, "sections": clean_sections}


def summarize_rows_for_ai(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for index, row in enumerate(rows, start=1):
        summary.append(
            {
                "rank": index,
                "image_label": row_label(row),
                "original_filename": row.get("original_filename") or row.get("filename"),
                "overall_score": row.get("overall_score"),
                "quality_score": row.get("quality_score"),
                "exposure": row.get("exposure"),
                "color_cast": row.get("color_cast"),
                "width": row.get("width"),
                "height": row.get("height"),
                "normalized_scores": row.get("normalized_scores"),
                "metrics": {
                    "brightness": row.get("brightness"),
                    "contrast": row.get("contrast"),
                    "sharpness": row.get("sharpness"),
                    "noise": row.get("noise"),
                    "saturation": row.get("saturation"),
                    "resolution_mp": row.get("resolution_mp"),
                    "dynamic_range": row.get("dynamic_range"),
                    "colorfulness": row.get("colorfulness"),
                    "entropy": row.get("entropy"),
                    "shadow_clip_pct": row.get("shadow_clip_pct"),
                    "highlight_clip_pct": row.get("highlight_clip_pct"),
                    "edge_density": row.get("edge_density"),
                    "texture_complexity": row.get("texture_complexity"),
                },
                "detected_counts": {
                    "text": row.get("text_count"),
                    "objects": row.get("object_count"),
                    "faces": row.get("face_count"),
                },
                "text_detected_summary": (row.get("ocr") or {}).get("summary"),
                "objects_detected_summary": (row.get("objects") or {}).get("summary"),
                "faces_detected_summary": (row.get("faces") or {}).get("summary"),
                "defects": row.get("defects"),
                "recommendations": row.get("recommendations"),
                "ranking_factors": row.get("ranking_factors"),
            }
        )
    return summary[:12]


def write_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "filename",
        "overall_score",
        "width",
        "height",
        "resolution_mp",
        "brightness",
        "contrast",
        "sharpness",
        "blur_score",
        "noise",
        "saturation",
        "exposure",
        "color_cast",
        "dynamic_range",
        "colorfulness",
        "entropy",
        "shadow_clip_pct",
        "highlight_clip_pct",
        "edge_density",
        "texture_complexity",
        "quality_score",
        "text_count",
        "object_count",
        "face_count",
        "defects",
        "recommendations",
    ]
    table = []
    for index, row in enumerate(rows, start=1):
        table.append({**row, "rank": index})

    sheet_rows = [fields]
    for row in table:
        sheet_rows.append([format_cell_value(row.get(field, "")) for field in fields])

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", XLSX_CONTENT_TYPES)
        workbook.writestr("_rels/.rels", XLSX_RELS)
        workbook.writestr("xl/workbook.xml", XLSX_WORKBOOK)
        workbook.writestr("xl/_rels/workbook.xml.rels", XLSX_WORKBOOK_RELS)
        workbook.writestr("xl/styles.xml", XLSX_STYLES)
        workbook.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(sheet_rows))


def ensure_xlsx(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    xlsx = run_dir / "report.xlsx"
    csv_path = run_dir / "report.csv"
    if not xlsx.exists() or (csv_path.exists() and csv_path.stat().st_mtime > xlsx.stat().st_mtime):
        write_xlsx(xlsx, rows)


def format_cell_value(value: Any) -> Any:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(str(item.get("name") or item.get("label") or item) for item in value[:8])
        return "; ".join(str(item) for item in value[:8])
    if isinstance(value, dict):
        return value.get("summary") or json.dumps(value, ensure_ascii=True)
    return value


def build_sheet_xml(rows: list[list[Any]]) -> str:
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{column_name(col_index)}{row_index}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                escaped = html.escape(str(value), quote=False)
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def write_analysis_image(image_path: Path, output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        write_legacy_analysis_image(image, output_path, row)
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    scores = row.get("normalized_scores") or {}
    labels = [
        "brightness_balance",
        "contrast",
        "sharpness",
        "noise_control",
        "saturation",
        "dynamic_range",
        "clipping_control",
        "resolution",
        "edge_detail",
        "texture_detail",
    ]
    score_values = [float(scores.get(label) or 0) for label in labels]
    channel_means = cv2.mean(image)[:3]
    color_means = [channel_means[2], channel_means[1], channel_means[0]]
    zones = [
        float(np.mean(gray <= 40) * 100),
        float(np.mean((gray > 40) & (gray < 215)) * 100),
        float(np.mean(gray >= 215) * 100),
    ]

    fig = plt.figure(figsize=(16, 18), dpi=140, facecolor="#f6f7f2", constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.05, 1.05, 0.85], width_ratios=[1, 1], hspace=0.28, wspace=0.24)
    fig.suptitle(
        f"{row_label(row)} quality dashboard | Overall {float(row.get('overall_score') or 0):.1f}",
        fontsize=21,
        fontweight="bold",
        color="#171815",
    )

    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(rgb)
    ax_image.set_title("Preview with detections", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_image.axis("off")
    width = max(float(row.get("width") or image.shape[1]), 1)
    height = max(float(row.get("height") or image.shape[0]), 1)
    for face in (row.get("faces") or {}).get("items", [])[:20]:
        x, y, w, h = [float(value) for value in face.get("box", [0, 0, 0, 0])]
        ax_image.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#f97316", linewidth=max(width / 900, 1.4)))
    for item in (row.get("objects") or {}).get("items", [])[:20]:
        x1, y1, x2, y2 = [float(value) for value in item.get("box", [0, 0, 0, 0])]
        ax_image.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#15803d", linewidth=max(width / 900, 1.4)))
    for text in (row.get("ocr") or {}).get("items", [])[:24]:
        x, y, w, h = [float(value) for value in text.get("box", [0, 0, 0, 0])]
        ax_image.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#2563eb", linewidth=max(width / 1200, 1.0)))
    ax_image.set_xlim(0, width)
    ax_image.set_ylim(height, 0)

    ax_scores = fig.add_subplot(grid[0, 1])
    y_pos = np.arange(len(labels))
    colors = ["#166534" if value >= 70 else "#b45309" if value >= 40 else "#b91c1c" for value in score_values]
    ax_scores.barh(y_pos, score_values, color=colors, alpha=0.92)
    ax_scores.set_yticks(y_pos, [label.replace("_", " ").title() for label in labels], fontsize=8)
    ax_scores.invert_yaxis()
    ax_scores.set_xlim(0, 100)
    ax_scores.set_title("Editor quality controls, normalized 0-100", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_scores.axvline(70, color="#6b7280", linestyle="--", linewidth=1)
    ax_scores.grid(axis="x", color="#d9ddd2", linewidth=0.8)
    for y, value in zip(y_pos, score_values):
        ax_scores.text(min(value + 1.4, 96), y, f"{value:.1f}", va="center", fontsize=8, color="#171815")

    ax_rgb = fig.add_subplot(grid[1, 0])
    for channel, color, name in [(0, "#dc2626", "Red"), (1, "#16a34a", "Green"), (2, "#2563eb", "Blue")]:
        hist, bins = np.histogram(rgb[:, :, channel].ravel(), bins=256, range=(0, 255), density=True)
        ax_rgb.plot(bins[:-1], hist, color=color, linewidth=1.7, label=name)
    ax_rgb.set_title("RGB channel distribution", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_rgb.set_xlabel("Pixel value")
    ax_rgb.set_ylabel("Density")
    ax_rgb.legend(frameon=False, fontsize=9)
    ax_rgb.grid(color="#d9ddd2", linewidth=0.8)

    ax_luma = fig.add_subplot(grid[1, 1])
    hist, bins = np.histogram(gray.ravel(), bins=256, range=(0, 255), density=True)
    ax_luma.fill_between(bins[:-1], hist, color="#4b5563", alpha=0.22)
    ax_luma.plot(bins[:-1], hist, color="#111827", linewidth=1.6)
    ax_luma.axvspan(0, 40, color="#1d4ed8", alpha=0.12, label="Shadows")
    ax_luma.axvspan(215, 255, color="#f59e0b", alpha=0.16, label="Highlights")
    ax_luma.set_title("Luminance and clipping risk", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_luma.set_xlabel("Brightness")
    ax_luma.set_ylabel("Density")
    ax_luma.legend(frameon=False, fontsize=9)
    ax_luma.grid(color="#d9ddd2", linewidth=0.8)

    ax_channels = fig.add_subplot(grid[2, 0])
    ax_channels.bar(["Red", "Green", "Blue"], color_means, color=["#dc2626", "#16a34a", "#2563eb"], alpha=0.86)
    ax_channels.set_ylim(0, 255)
    ax_channels.set_title("Average color balance", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_channels.set_ylabel("Mean channel value")
    ax_channels.grid(axis="y", color="#d9ddd2", linewidth=0.8)
    cast = str(row.get("color_cast") or "neutral")
    ax_channels.text(0.02, 0.92, f"Detected cast: {cast}", transform=ax_channels.transAxes, fontsize=11, fontweight="bold")

    ax_zones = fig.add_subplot(grid[2, 1])
    ax_zones.bar(["Shadows", "Midtones", "Highlights"], zones, color=["#1d4ed8", "#64748b", "#f59e0b"], alpha=0.86)
    ax_zones.set_ylim(0, 100)
    ax_zones.set_title("Tone zones by pixel share", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_zones.set_ylabel("% of image")
    ax_zones.grid(axis="y", color="#d9ddd2", linewidth=0.8)
    for index, value in enumerate(zones):
        ax_zones.text(index, value + 1.2, f"{value:.1f}%", ha="center", fontsize=9)

    ax_notes = fig.add_subplot(grid[3, :])
    ax_notes.axis("off")
    note_lines = [
        f"Resolution: {row.get('width')} x {row.get('height')} ({row.get('resolution_mp')} MP)",
        f"Exposure: {row.get('exposure')} | Brightness: {row.get('brightness')} | Contrast: {row.get('contrast')}",
        f"Sharpness: {row.get('sharpness')} | Noise: {row.get('noise')} | Saturation: {row.get('saturation')}",
        f"Any text detected: {row.get('text_count') or 0} | Faces detected: {row.get('face_count') or 0} | Objects detected: {row.get('object_count') or 0}",
    ]
    defects = row.get("defects") or []
    recommendations = row.get("recommendations") or []
    note_lines.append("")
    note_lines.append("Top editor actions:")
    if recommendations:
        note_lines.extend(f"- {item}" for item in recommendations[:5])
    elif defects:
        note_lines.extend(f"- Review {defect.get('name', 'detected defect')}" for defect in defects[:5])
    else:
        note_lines.append("- No urgent correction detected.")
    wrapped = "\n".join("\n".join(textwrap.wrap(line, 112)) if line and not line.startswith("- ") else line for line in note_lines)
    ax_notes.text(
        0,
        1,
        wrapped,
        va="top",
        ha="left",
        fontsize=10.2,
        color="#1f2937",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#ffffff", "edgecolor": "#d7dbd0"},
    )

    for ax in fig.axes:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#d7dbd0")

    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_legacy_analysis_image(image: np.ndarray, output_path: Path, row: dict[str, Any]) -> None:
    canvas = np.full((720, 1100, 3), 250, dtype=np.uint8)
    preview = fit_image(image, 520, 330)
    y0, x0 = 76, 28
    canvas[y0 : y0 + preview.shape[0], x0 : x0 + preview.shape[1]] = preview
    scale = preview.shape[1] / image.shape[1]
    roi = canvas[y0 : y0 + preview.shape[0], x0 : x0 + preview.shape[1]]
    draw_feature_boxes(roi, row, scale)
    cv2.putText(canvas, "Image quality analysis", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 28, 22), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Score {row['overall_score']} | {row['exposure']} | {row['color_cast']} cast", (28, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (82, 88, 76), 1, cv2.LINE_AA)
    draw_score_bars(canvas, row.get("normalized_scores") or {}, 585, 80)
    rgb_hist = histogram_panel(image, "RGB histogram", color=True)
    gray_hist = histogram_panel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), "Luma histogram", color=False)
    canvas[430:650, 28:548] = rgb_hist
    canvas[430:650, 580:1100] = gray_hist
    cv2.imwrite(str(output_path), canvas)


def fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    panel = np.full((max_height, max_width, 3), 238, dtype=np.uint8)
    y = (max_height - resized.shape[0]) // 2
    x = (max_width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def draw_score_bars(canvas: np.ndarray, scores: dict[str, Any], x: int, y: int) -> None:
    labels = ["brightness_balance", "contrast", "sharpness", "noise_control", "dynamic_range", "clipping_control"]
    for index, key in enumerate(labels):
        value = float(scores.get(key) or 0)
        top = y + index * 44
        cv2.putText(canvas, key.replace("_", " ").title(), (x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 75, 66), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, top + 12), (x + 380, top + 24), (226, 232, 221), -1)
        cv2.rectangle(canvas, (x, top + 12), (x + int(380 * clamp(value) / 100), top + 24), (70, 106, 81), -1)
        cv2.putText(canvas, f"{value:.1f}", (x + 396, top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (24, 28, 22), 1, cv2.LINE_AA)


def histogram_panel(image: np.ndarray, title: str, color: bool) -> np.ndarray:
    panel = np.full((220, 520, 3), 255, dtype=np.uint8)
    cv2.putText(panel, title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (24, 28, 22), 1, cv2.LINE_AA)
    plot = np.full((160, 480, 3), 248, dtype=np.uint8)
    if color:
        channels = [(0, (180, 75, 75)), (1, (70, 150, 80)), (2, (70, 95, 190))]
        for channel, line_color in channels:
            hist = cv2.calcHist([image], [channel], None, [256], [0, 256]).ravel()
            draw_hist_line(plot, hist, line_color)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        draw_hist_line(plot, hist, (110, 110, 110))
    panel[44:204, 20:500] = plot
    return panel


def draw_hist_line(plot: np.ndarray, hist: np.ndarray, color: tuple[int, int, int]) -> None:
    hist = hist / max(float(hist.max()), 1.0)
    height, width = plot.shape[:2]
    points = []
    for index, value in enumerate(hist):
        x = int(index / 255 * (width - 1))
        y = int((1 - value) * (height - 1))
        points.append((x, y))
    for start, end in zip(points, points[1:]):
        cv2.line(plot, start, end, color, 1, cv2.LINE_AA)


def draw_feature_boxes(image: np.ndarray, row: dict[str, Any], scale: float) -> None:
    for face in (row.get("faces") or {}).get("items", [])[:10]:
        x, y, w, h = [int(value * scale) for value in face.get("box", [0, 0, 0, 0])]
        cv2.rectangle(image, (x, y), (x + w, y + h), (54, 119, 255), 2)
        cv2.putText(image, "face", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (54, 119, 255), 1, cv2.LINE_AA)
    for item in (row.get("objects") or {}).get("items", [])[:10]:
        x1, y1, x2, y2 = [int(value * scale) for value in item.get("box", [0, 0, 0, 0])]
        cv2.rectangle(image, (x1, y1), (x2, y2), (28, 153, 84), 2)
        cv2.putText(image, str(item.get("label", "object"))[:18], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (28, 153, 84), 1, cv2.LINE_AA)
    for text in (row.get("ocr") or {}).get("items", [])[:14]:
        x, y, w, h = [int(value * scale) for value in text.get("box", [0, 0, 0, 0])]
        cv2.rectangle(image, (x, y), (x + w, y + h), (194, 109, 29), 1)


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        write_legacy_comparison(path, rows)
        return

    available_rows = [row for row in rows if cv2.imread(str(row.get("filepath", ""))) is not None]
    if not available_rows:
        return
    labels = [f"Image {index}" for index, _ in enumerate(available_rows, start=1)]
    x = np.arange(len(available_rows))
    overall = [float(row.get("overall_score") or 0) for row in available_rows]
    quality = [float(row.get("quality_score") or row.get("overall_score") or 0) for row in available_rows]
    editor_metrics = [
        "brightness_balance",
        "contrast",
        "sharpness",
        "noise_control",
        "saturation",
        "dynamic_range",
        "clipping_control",
        "resolution",
        "edge_detail",
        "texture_detail",
    ]
    metric_matrix = np.array(
        [[float((row.get("normalized_scores") or {}).get(metric) or 0) for row in available_rows] for metric in editor_metrics],
        dtype=float,
    )

    fig = plt.figure(figsize=(19, 12), dpi=140, facecolor="#f6f7f2")
    grid = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.05, 1.05], width_ratios=[1, 1, 1], hspace=0.38, wspace=0.28)
    fig.suptitle("All-image quality comparison dashboard", fontsize=22, fontweight="bold", y=0.98, color="#171815")

    thumb_grid = grid[0, :].subgridspec(1, len(available_rows), wspace=0.18)
    for index, row in enumerate(available_rows):
        ax = fig.add_subplot(thumb_grid[0, index])
        image = cv2.imread(str(row["filepath"]))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{labels[index]} | {overall[index]:.1f}", fontsize=11, fontweight="bold")
        ax.axis("off")

    ax_scores = fig.add_subplot(grid[1, 0:2])
    width = 0.32
    ax_scores.bar(x - width / 2, overall, width, label="Overall", color="#166534")
    ax_scores.bar(x + width / 2, quality, width, label="Quality", color="#2563eb")
    ax_scores.set_xticks(x, labels)
    ax_scores.set_ylim(0, 100)
    ax_scores.set_ylabel("Score")
    ax_scores.set_title("Quality ranking scores side by side", loc="left", fontsize=13, fontweight="bold")
    ax_scores.legend(frameon=False)
    ax_scores.grid(axis="y", color="#d9ddd2", linewidth=0.8)

    ax_heat = fig.add_subplot(grid[2, 0:2])
    heat = ax_heat.imshow(metric_matrix, aspect="auto", cmap="RdYlGn")
    ax_heat.set_yticks(np.arange(len(editor_metrics)), [metric.replace("_", " ").title() for metric in editor_metrics], fontsize=9)
    ax_heat.set_xticks(x, labels)
    ax_heat.set_title("Editor controls, 0-100 where green means stronger", loc="left", fontsize=13, fontweight="bold")
    for y in range(metric_matrix.shape[0]):
        for col in range(metric_matrix.shape[1]):
            ax_heat.text(col, y, f"{metric_matrix[y, col]:.0f}", ha="center", va="center", fontsize=8, color="#111827")
    fig.colorbar(heat, ax=ax_heat, fraction=0.028, pad=0.02, label="Relative strength")

    ax_defects = fig.add_subplot(grid[1, 2])
    defect_counts: dict[str, int] = {}
    for row in available_rows:
        for defect in row.get("defects") or []:
            name = str(defect.get("name") or "Defect")
            defect_counts[name] = defect_counts.get(name, 0) + 1
    if defect_counts:
        names = list(defect_counts.keys())[:8]
        counts = [defect_counts[name] for name in names]
        y_pos = np.arange(len(names))
        ax_defects.barh(y_pos, counts, color="#b45309", alpha=0.86)
        ax_defects.set_yticks(y_pos, names)
        ax_defects.invert_yaxis()
        ax_defects.set_xlim(0, max(len(available_rows), max(counts)) + 0.35)
        ax_defects.set_xticks(range(0, len(available_rows) + 1))
        ax_defects.set_xlabel("Number of uploaded images")
        ax_defects.tick_params(axis="y", labelsize=8.5)
        for y, count in zip(y_pos, counts):
            ax_defects.text(
                count + 0.04,
                y,
                f"{count} of {len(available_rows)}",
                va="center",
                fontsize=9,
                color="#1f2937",
                fontweight="bold",
            )
    else:
        ax_defects.text(0.5, 0.5, "No major defects detected", ha="center", va="center", fontsize=12)
    ax_defects.set_title("Defect frequency across uploaded images", loc="left", fontsize=12.2, fontweight="bold")
    ax_defects.text(
        0,
        1.04,
        "Longer bar = this issue appears in more images",
        transform=ax_defects.transAxes,
        fontsize=8.5,
        color="#4b5563",
    )
    ax_defects.grid(axis="x", color="#d9ddd2", linewidth=0.8)

    ax_actions = fig.add_subplot(grid[2, 2])
    ax_actions.axis("off")
    action_lines = ["Most useful editing priorities:"]
    worst_noise = max(available_rows, key=lambda row: float(row.get("noise") or 0))
    lowest_sharp = min(available_rows, key=lambda row: float(row.get("sharpness") or 0))
    highest_clip = max(available_rows, key=lambda row: float(row.get("highlight_clip_pct") or 0) + float(row.get("shadow_clip_pct") or 0))
    action_lines.extend(
        [
            f"- Cleanest target: {labels[overall.index(max(overall))]} has the best overall score.",
            f"- Noise watch: Image {available_rows.index(worst_noise) + 1} has the highest noise ({worst_noise.get('noise')}).",
            f"- Sharpness watch: Image {available_rows.index(lowest_sharp) + 1} has the weakest sharpness ({lowest_sharp.get('sharpness')}).",
            f"- Clipping watch: Image {available_rows.index(highest_clip) + 1} has the strongest clipping risk.",
        ]
    )
    wrapped = "\n".join("\n".join(textwrap.wrap(line, 58)) if not line.startswith("- ") else line for line in action_lines)
    ax_actions.text(
        0,
        1,
        wrapped,
        va="top",
        ha="left",
        fontsize=11,
        color="#1f2937",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#ffffff", "edgecolor": "#d7dbd0"},
    )

    for ax in fig.axes:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#d7dbd0")

    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def normalize_metric_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(matrix, dtype=float)
    for index, row in enumerate(matrix):
        low = float(np.min(row))
        high = float(np.max(row))
        if math.isclose(high, low):
            normalized[index] = 50
        else:
            normalized[index] = (row - low) / (high - low) * 100
    return normalized


def write_legacy_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    thumbs = []
    for index, row in enumerate(rows, start=1):
        image = cv2.imread(str(row["filepath"]))
        if image is None:
            continue
        image = cv2.resize(image, (260, 160), interpolation=cv2.INTER_AREA)
        panel = np.full((220, 260, 3), 245, dtype=np.uint8)
        panel[:160, :] = image
        text = f"#{index} {row['overall_score']}"
        cv2.putText(panel, text, (12, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        name = str(row["filename"])[:28]
        cv2.putText(panel, name, (12, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        thumbs.append(panel)
    if thumbs:
        comparison = np.hstack(thumbs)
        cv2.imwrite(str(path), comparison)


XLSX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

XLSX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

XLSX_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Image Quality Report" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""

XLSX_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

XLSX_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>"""
