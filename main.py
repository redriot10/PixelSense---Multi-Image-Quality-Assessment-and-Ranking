from core.metrics import analyze_image, export_report

IMAGE_PATH = r"C:\RED's space\Wallpapers\valorant.jpg"

result = analyze_image(IMAGE_PATH, save_chart=True)

# Export CSV report
export_report(result)

print("\n" + "=" * 50)
print("  VISIONSCORE — ANALYSIS RESULT")
print("=" * 50)
print(f"  File          : {result.filepath}")
print(f"  Overall Score : {result.overall_score} / 100")
print(f"  Brightness    : {result.brightness}")
print(f"  Contrast      : {result.contrast}")
print(f"  Sharpness     : {result.sharpness}")
print(f"  Blur Score    : {result.blur_score}  (0=sharp, 1=blurry)")
print(f"  Noise         : {result.noise}")
print(f"  Saturation    : {result.saturation}")
print(f"  Exposure      : {result.exposure}")
print(f"  Color Cast    : {result.color_cast}")
print(f"  Resolution    : {result.resolution_mp} MP")
print(f"  Dynamic Range : {result.dynamic_range}")
print("=" * 50)