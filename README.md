# PixelSense

PixelSense is an AI-assisted image quality platform for comparing many uploaded images, ranking them by visual quality, detecting common defects, and exporting clean analysis reports.

## Features

- Upload and analyze multiple images in one run
- Rank images by overall quality score
- Measure brightness, contrast, sharpness, noise, saturation, resolution, dynamic range, clipping, edge density, and texture detail
- Detect blur, underexposure, overexposure, low contrast, noise, clipping, and color cast
- Generate annotated visual dashboards for every image
- Export CSV and Excel reports
- Optional OCR with Tesseract
- Optional object detection with YOLO
- Optional MongoDB persistence for analysis runs
- Chat assistant powered by Groq with local fallback responses
- React frontend served by the FastAPI backend
- Docker-ready for local testing and container hosting

## Tech Stack

- Frontend: React, Vite
- Backend: FastAPI, Uvicorn
- Image processing: OpenCV, NumPy, Matplotlib, Pillow
- OCR: Tesseract, pytesseract
- Object detection: Ultralytics YOLO
- Database: MongoDB
- AI chat: Groq API
- Deployment: Docker-compatible hosting

## Environment Variables

Create a `.env` file from `.env.example`:

```env
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

MONGODB_URI=
MONGODB_DATABASE=PIXELSENSE
MONGODB_COLLECTION=image_metrics
```

`GROQ_API_KEY` and `MONGODB_URI` are optional for basic local image analysis, but the chat and database features need them.

## Local Development

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install and build the frontend:

```bash
cd frontend
npm install
npm run build
```

Run the backend:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

For frontend-only development:

```bash
cd frontend
npm run dev
```

## Run With Docker

Build the image from the project root:

```bash
docker build -t pixelsense .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 pixelsense
```

Open:

```text
http://localhost:8000
```

Use a local volume if you want uploaded files and generated reports to survive container restarts:

```bash
docker run --env-file .env -p 8000:8000 -v ./runtime-data:/app/runtime-data pixelsense
```

## Deployment

PixelSense is packaged with a production `Dockerfile`. Deploy it on any host that supports Docker containers.

Recommended runtime environment variables:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=PIXELSENSE
MONGODB_COLLECTION=image_metrics
APP_DATA_DIR=/app/runtime-data
MODEL_DIR=/app/data/models
```

For long-term uploaded files and generated reports, attach persistent storage at:

```text
/app/runtime-data
```

## Health Check

The backend exposes:

```text
/health
```

Use this endpoint to verify that the app, OCR, YOLO, Groq configuration, and MongoDB connection are available.
