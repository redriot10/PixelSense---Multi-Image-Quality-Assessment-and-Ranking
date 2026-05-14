# Multi-Image Quality Analysis and Ranking System

An AI-assisted image quality workbench that analyzes multiple uploaded images, ranks them by visual quality, detects common defects, generates reports, and provides conversational explanations using Groq.

## Features

- Upload and analyze multiple images at once
- Rank images by overall quality score
- Measure brightness, contrast, sharpness, noise, saturation, resolution, dynamic range, clipping, edge density, and texture detail
- Detect defects such as blur, underexposure, overexposure, low contrast, noise, clipping, and color cast
- Generate annotated visual analysis dashboards
- Export CSV and Excel reports
- Optional OCR with Tesseract
- Optional object detection with YOLO
- Optional MongoDB persistence for analysis runs
- Chat assistant powered by Groq with local fallback responses
- React frontend served by FastAPI backend
- Docker and Railway deployment ready

## Tech Stack

- Frontend: React, Vite
- Backend: FastAPI, Uvicorn
- Image processing: OpenCV, NumPy, Matplotlib, Pillow
- OCR: Tesseract, pytesseract
- Object detection: Ultralytics YOLO
- Database: MongoDB
- AI chat: Groq API
- Deployment: Docker, Railway

## Local Setup

Create a `.env` file from `.env.example`:

```env
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

MONGODB_URI=
MONGODB_DATABASE=PIXELSENSE
MONGODB_COLLECTION=image_metrics
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Build the frontend:

```bash
npm run build
```

Run the backend:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8010
```

For frontend development:

```bash
cd frontend
npm run dev
```

## Deployment

This project is ready for Railway deployment using the included `Dockerfile` and `railway.json`.

Recommended Railway environment variables:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=PIXELSENSE
MONGODB_COLLECTION=image_metrics
APP_DATA_DIR=/app/runtime-data
MODEL_DIR=/app/data/models
```

For persistent uploaded files and generated reports, add a Railway volume mounted at:

```text
/app/runtime-data
```

## Health Check

The backend exposes:

```text
/health
```

Railway uses this endpoint to verify successful deployments.

