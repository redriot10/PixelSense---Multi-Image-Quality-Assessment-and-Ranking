# Advanced Vision Models

This folder is for local model files. The backend never downloads model weights automatically.

Supported object detection options:

- `data/models/yolov8n.pt` with the `ultralytics` package installed.
- `data/models/yolo/yolov4-tiny.cfg`
- `data/models/yolo/yolov4-tiny.weights`
- `data/models/yolo/coco.names`

OCR requires:

- the `pytesseract` Python package
- the Tesseract OCR desktop binary installed and available on `PATH`

Face detection uses OpenCV's bundled Haar cascade and should work without extra files.
