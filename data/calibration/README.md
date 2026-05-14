# Calibration Set

Use this folder to tune the image-quality logic with real examples.

## How to use

1. Put sample images in `data/calibration/images/`.
2. Edit `data/calibration/truth.json` with the expected human labels.
3. Run:

```powershell
.\.venv\Scripts\python.exe -B tools\calibrate.py
```

The script writes a JSON report to `data/calibration/results/latest.json` and prints a short pass/fail summary.

## Suggested labels

- `faces`: expected number of real human faces.
- `text_present`: whether useful readable text is present.
- `blur`: one of `sharp`, `usable`, `soft`, `blurry`, `out_of_focus`.
- `exposure`: one of `underexposed`, `normal`, `overexposed`.
- `noise`: one of `low`, `medium`, `high`.

Start with 20-50 real images from your target use case. Add tricky examples too: printed face icons, curved labels, dark photos, bright photos, and true blurry photos.
