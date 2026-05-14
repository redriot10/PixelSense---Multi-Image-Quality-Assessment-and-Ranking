from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import score_image

CALIBRATION_DIR = ROOT / "data" / "calibration"
IMAGE_DIR = CALIBRATION_DIR / "images"
RESULT_DIR = CALIBRATION_DIR / "results"
TRUTH_PATH = CALIBRATION_DIR / "truth.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    seed_sample_image()

    truth = load_truth()
    cases = truth.get("images", [])
    if not cases:
        print("No calibration cases found in data/calibration/truth.json")
        return

    results = []
    total_checks = 0
    failed_checks = 0
    for case in cases:
        result = evaluate_case(case)
        results.append(result)
        total_checks += len(result["checks"])
        failed_checks += sum(1 for check in result["checks"] if not check["pass"])

    report = {
        "summary": {
            "cases": len(results),
            "checks": total_checks,
            "passed": total_checks - failed_checks,
            "failed": failed_checks,
            "pass_rate": round(((total_checks - failed_checks) / max(total_checks, 1)) * 100, 1),
        },
        "results": results,
    }
    output_path = RESULT_DIR / "latest.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Calibration cases: {summary['cases']}")
    print(f"Checks passed: {summary['passed']}/{summary['checks']} ({summary['pass_rate']}%)")
    if failed_checks:
        print("Failures:")
        for result in results:
            for check in result["checks"]:
                if not check["pass"]:
                    print(f"- {result['file']}: {check['name']} expected {check['expected']!r}, got {check['actual']!r}")
    print(f"Report written: {output_path}")


def seed_sample_image() -> None:
    sample_name = "WhatsApp Image 2026-05-10 at 6.57.37 PM.jpeg"
    target = IMAGE_DIR / sample_name
    if target.exists():
        return
    candidates = [
        ROOT / "data" / "input" / "uploads" / "d4aa27b741e54371987bb614674433f4" / "WhatsApp Image 2026-05-10 at 6.57.37 PM_5f21547a.jpeg",
        Path.home() / "Downloads" / sample_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, target)
            return


def load_truth() -> dict[str, Any]:
    if not TRUTH_PATH.exists():
        return {"images": []}
    return json.loads(TRUTH_PATH.read_text(encoding="utf-8"))


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    file_name = str(case.get("file", "")).strip()
    image_path = IMAGE_DIR / file_name
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.exists():
        return {
            "file": file_name,
            "error": "Image file is missing from data/calibration/images.",
            "checks": [{"name": "file_exists", "expected": True, "actual": False, "pass": False}],
        }

    metrics = score_image(image_path)
    predicted = classify_metrics(metrics)
    checks = []
    for key in ("faces", "text_present", "blur", "exposure", "noise"):
        if key not in case:
            continue
        checks.append(
            {
                "name": key,
                "expected": case[key],
                "actual": predicted[key],
                "pass": case[key] == predicted[key],
            }
        )

    return {
        "file": file_name,
        "notes": case.get("notes", ""),
        "checks": checks,
        "predicted": predicted,
        "metrics": {
            "brightness": metrics["brightness"],
            "contrast": metrics["contrast"],
            "sharpness": metrics["sharpness"],
            "blur_score": metrics["blur_score"],
            "noise": metrics["noise"],
            "quality_score": metrics["quality_score"],
            "text_count": metrics["text_count"],
            "face_count": metrics["face_count"],
            "object_count": metrics["object_count"],
            "defects": metrics["defects"],
        },
    }


def classify_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    text_count = int(metrics.get("text_count") or 0)
    sharpness = float(metrics.get("sharpness") or 0)
    noise = float(metrics.get("noise") or 0)
    return {
        "faces": int(metrics.get("face_count") or 0),
        "text_present": text_count > 0,
        "blur": classify_blur(sharpness),
        "exposure": str(metrics.get("exposure") or "normal"),
        "noise": classify_noise(noise),
    }


def classify_blur(sharpness: float) -> str:
    if sharpness < 60:
        return "out_of_focus"
    if sharpness < 110:
        return "blurry"
    if sharpness < 150:
        return "usable"
    if sharpness < 260:
        return "sharp"
    return "sharp"


def classify_noise(noise: float) -> str:
    if noise > 18:
        return "high"
    if noise > 9:
        return "medium"
    return "low"


if __name__ == "__main__":
    main()
