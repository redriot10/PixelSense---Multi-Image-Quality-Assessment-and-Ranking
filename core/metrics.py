import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


# ─────────────────────────────────────────
# RESULT CONTAINER
# One ImageMetrics object holds ALL results
# for a single image — clean, organized
# ─────────────────────────────────────────
@dataclass
class ImageMetrics:
    filepath:       str
    brightness:     float
    contrast:       float
    sharpness:      float
    blur_score:     float
    noise:          float
    saturation:     float
    exposure:       str
    color_cast:     str
    resolution_mp:  float
    dynamic_range:  float
    overall_score:  float


# ─────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────
def load_image(filepath: str):
    bgr = cv2.imread(filepath)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {filepath}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


# ─────────────────────────────────────────
# METRIC 1 — BRIGHTNESS
# Mean of all grayscale pixel values
# 0 = black,  255 = white,  128 = ideal
# ─────────────────────────────────────────
def compute_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


# ─────────────────────────────────────────
# METRIC 2 — CONTRAST
# Std deviation of grayscale pixels
# Low std = flat grey image
# High std = wide tonal range
# ─────────────────────────────────────────
def compute_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


# ─────────────────────────────────────────
# METRIC 3 — SHARPNESS
# Laplacian detects edges
# Variance of Laplacian = edge strength
# Low = blurry,  High = sharp
# ─────────────────────────────────────────
def compute_sharpness(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


# ─────────────────────────────────────────
# METRIC 4 — BLUR SCORE  (0=sharp, 1=blurry)
# Normalized inverse of sharpness
# ─────────────────────────────────────────
def compute_blur_score(sharpness: float) -> float:
    normalized = sharpness / 1000.0
    return float(np.clip(1.0 - normalized, 0.0, 1.0))


# ─────────────────────────────────────────
# METRIC 5 — NOISE
# Blur the image, subtract from original
# What remains = noise floor
# ─────────────────────────────────────────
def compute_noise(gray: np.ndarray) -> float:
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.std(residual))


# ─────────────────────────────────────────
# METRIC 6 — SATURATION
# Convert to HSV, read S channel
# 0 = grey/colorless,  255 = vivid
# ─────────────────────────────────────────
def compute_saturation(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


# ─────────────────────────────────────────
# METRIC 7 — EXPOSURE LABEL
# Classify brightness into 3 buckets
# ─────────────────────────────────────────
def classify_exposure(brightness: float) -> str:
    if brightness < 60:
        return "underexposed"
    elif brightness > 200:
        return "overexposed"
    return "normal"


# ─────────────────────────────────────────
# METRIC 8 — COLOR CAST
# Compare mean R vs mean B channel
# Warm = too red/yellow,  Cool = too blue
# ─────────────────────────────────────────
def detect_color_cast(bgr: np.ndarray) -> str:
    b = float(np.mean(bgr[:, :, 0]))
    r = float(np.mean(bgr[:, :, 2]))
    diff = r - b
    if diff > 15:
        return "warm"
    elif diff < -15:
        return "cool"
    return "neutral"


# ─────────────────────────────────────────
# METRIC 9 — RESOLUTION
# Total pixels converted to megapixels
# ─────────────────────────────────────────
def compute_resolution(bgr: np.ndarray) -> float:
    h, w = bgr.shape[:2]
    return round((h * w) / 1_000_000, 2)


# ─────────────────────────────────────────
# METRIC 10 — DYNAMIC RANGE
# max pixel - min pixel in grayscale
# 255 = full tonal range used
# ─────────────────────────────────────────
def compute_dynamic_range(gray: np.ndarray) -> float:
    return float(np.max(gray) - np.min(gray))


# ─────────────────────────────────────────
# METRIC 11 — OVERALL SCORE  (0–100)
# Weighted combination of all signals
# ─────────────────────────────────────────
def compute_overall_score(sharpness, contrast, brightness,
                           saturation, dynamic_range, noise) -> float:
    sharp_n    = float(np.clip(sharpness / 500.0,      0, 1))
    contrast_n = float(np.clip(contrast / 80.0,        0, 1))
    bright_n   = 1.0 - abs(brightness - 128) / 128.0
    sat_n      = float(np.clip(saturation / 150.0,     0, 1))
    dr_n       = float(np.clip(dynamic_range / 200.0,  0, 1))
    noise_n    = float(np.clip(1.0 - noise / 20.0,     0, 1))

    score = (
        sharp_n    * 0.30 +
        contrast_n * 0.20 +
        bright_n   * 0.20 +
        sat_n      * 0.10 +
        dr_n       * 0.10 +
        noise_n    * 0.10
    )
    return round(score * 100, 1)


# ─────────────────────────────────────────
# VISUALIZATION — saves analysis chart
# to data/output/
# ─────────────────────────────────────────
def plot_analysis(bgr: np.ndarray, gray: np.ndarray,
                  metrics: ImageMetrics, save_path: str):

    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"VisionScore — {metrics.filepath}", fontsize=13)

    # Top-left: original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    # Top-right: RGB histogram
    for i, color in enumerate(["blue", "green", "red"]):
        axes[0, 1].hist(bgr[:, :, i].flatten(),
                        bins=128, color=color, alpha=0.5, label=color.upper())
    axes[0, 1].set_title("RGB histogram")
    axes[0, 1].set_xlabel("Pixel value")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # Bottom-left: quality metric bar chart
    names  = ["Brightness", "Contrast", "Sharpness", "Saturation", "Noise"]
    values = [
        round(metrics.brightness / 2.55, 1),   # normalize to 0-100
        round(metrics.contrast / 1.28, 1),
        round(min(metrics.sharpness / 5.0, 100), 1),
        round(metrics.saturation / 2.55, 1),
        round(max(0, 100 - metrics.noise * 5), 1),
    ]
    colors = ["#2ecc71" if v >= 60 else "#e74c3c" for v in values]
    axes[1, 0].barh(names, values, color=colors)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].axvline(x=60, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_title(f"Quality metrics  |  Overall: {metrics.overall_score}/100")

    # Bottom-right: grayscale histogram
    axes[1, 1].hist(gray.flatten(), bins=256, color="gray", alpha=0.8)
    axes[1, 1].set_title("Grayscale histogram")
    axes[1, 1].set_xlabel("Pixel value (0=black, 255=white)")
    axes[1, 1].set_ylabel("Pixel count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────
# MASTER FUNCTION
# Call this with any image path
# Returns a complete ImageMetrics object
# ─────────────────────────────────────────
def analyze_image(filepath: str, save_chart: bool = True) -> ImageMetrics:

    bgr, gray = load_image(filepath)

    brightness    = compute_brightness(gray)
    contrast      = compute_contrast(gray)
    sharpness     = compute_sharpness(gray)
    blur_score    = compute_blur_score(sharpness)
    noise         = compute_noise(gray)
    saturation    = compute_saturation(bgr)
    dynamic_range = compute_dynamic_range(gray)
    resolution    = compute_resolution(bgr)
    exposure      = classify_exposure(brightness)
    color_cast    = detect_color_cast(bgr)
    overall       = compute_overall_score(sharpness, contrast, brightness,
                                          saturation, dynamic_range, noise)

    result = ImageMetrics(
        filepath      = filepath,
        brightness    = round(brightness, 2),
        contrast      = round(contrast, 2),
        sharpness     = round(sharpness, 2),
        blur_score    = round(blur_score, 3),
        noise         = round(noise, 2),
        saturation    = round(saturation, 2),
        exposure      = exposure,
        color_cast    = color_cast,
        resolution_mp = resolution,
        dynamic_range = round(dynamic_range, 2),
        overall_score = overall,
    )

    if save_chart:
        import os
        os.makedirs("data/output", exist_ok=True)
        chart_path = "data/output/analysis.png"
        plot_analysis(bgr, gray, result, chart_path)

    return result
 
 

def export_report(metrics: ImageMetrics, save_dir: str = "data/output") -> str:
    """
    Export metrics to a CSV report using Pandas.
    Returns the path of the saved file.
    """
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # Convert dataclass to dictionary, then to DataFrame
    data = {
        "metric": [
            "filepath", "brightness", "contrast", "sharpness",
            "blur_score", "noise", "saturation", "exposure",
            "color_cast", "resolution_mp", "dynamic_range", "overall_score"
        ],
        "value": [
            metrics.filepath, metrics.brightness, metrics.contrast,
            metrics.sharpness, metrics.blur_score, metrics.noise,
            metrics.saturation, metrics.exposure, metrics.color_cast,
            metrics.resolution_mp, metrics.dynamic_range, metrics.overall_score
        ]
    }

    df = pd.DataFrame(data)

    save_path = os.path.join(save_dir, "report.csv")
    df.to_csv(save_path, index=False)
    print(f"  Report saved → {save_path}")
    return save_path