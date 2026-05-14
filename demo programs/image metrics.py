import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# SECTION 1: Load image and basic info
# ─────────────────────────────────────────
img = cv2.imread(r"C:\RED's space\Wallpapers\valorant.jpg")

if img is None:
    print("ERROR: Image not found. Check the path.")
    exit()

print("=" * 50)
print("SECTION 1 — Basic image info")
print("=" * 50)
print(f"Type        : {type(img)}")
print(f"Shape       : {img.shape}")       # (height, width, 3)
print(f"dtype       : {img.dtype}")       # uint8 = 0 to 255
print(f"Total pixels: {img.shape[0] * img.shape[1]:,}")
print(f"Min value   : {img.min()}")
print(f"Max value   : {img.max()}")
print(f"Mean value  : {img.mean():.2f}")

# ─────────────────────────────────────────
# SECTION 2: Individual pixel access
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SECTION 2 — Pixel access")
print("=" * 50)

pixel_top_left  = img[0, 0]         # top-left corner
pixel_center    = img[img.shape[0]//2, img.shape[1]//2]  # exact center

print(f"Top-left pixel [B,G,R] : {pixel_top_left}")
print(f"Center pixel   [B,G,R] : {pixel_center}")

# Access individual channels of center pixel
B, G, R = pixel_center[0], pixel_center[1], pixel_center[2]
print(f"  → Blue  : {B}")
print(f"  → Green : {G}")
print(f"  → Red   : {R}")

# ─────────────────────────────────────────
# SECTION 3: Channel separation
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SECTION 3 — Channel separation (BGR)")
print("=" * 50)

blue_channel  = img[:, :, 0]   # all rows, all cols, channel 0
green_channel = img[:, :, 1]
red_channel   = img[:, :, 2]

print(f"Blue  channel — shape: {blue_channel.shape}  mean: {blue_channel.mean():.2f}")
print(f"Green channel — shape: {green_channel.shape}  mean: {green_channel.mean():.2f}")
print(f"Red   channel — shape: {red_channel.shape}  mean: {red_channel.mean():.2f}")

# Which color dominates this image?
means = {'Blue': blue_channel.mean(), 'Green': green_channel.mean(), 'Red': red_channel.mean()}
dominant = max(means, key=means.get)
print(f"\nDominant color: {dominant}  ← this tells us the color cast")

# ─────────────────────────────────────────
# SECTION 4: Grayscale conversion
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SECTION 4 — Grayscale conversion")
print("=" * 50)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray shape  : {gray.shape}")       # (height, width) — no channel axis
print(f"Gray dtype  : {gray.dtype}")
print(f"Gray min    : {gray.min()}")
print(f"Gray max    : {gray.max()}")
print(f"Gray mean   : {gray.mean():.2f}   ← this IS your brightness score")
print(f"Gray std    : {gray.std():.2f}    ← this IS your contrast score")

# ─────────────────────────────────────────
# SECTION 5: Cropping (NumPy slicing)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SECTION 5 — Cropping via NumPy slice")
print("=" * 50)

h, w = img.shape[:2]
# Crop center 100x100 patch
cy, cx = h // 2, w // 2
patch = img[cy-50 : cy+50, cx-50 : cx+50]   # [row_start:row_end, col_start:col_end]
print(f"Original shape : {img.shape}")
print(f"Cropped patch  : {patch.shape}")     # (100, 100, 3)
print(f"Patch mean     : {patch.mean():.2f}")

# ─────────────────────────────────────────
# SECTION 6: Image as Pandas DataFrame
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SECTION 6 — Image as Pandas DataFrame")
print("=" * 50)

# Use a small 50x50 sample (full image would be millions of rows)
sample = img[:50, :50]                        # top-left 50x50 pixels
sh, sw = sample.shape[:2]
sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

rows_idx, cols_idx = np.mgrid[0:sh, 0:sw]

df = pd.DataFrame({
    "row"       : rows_idx.flatten(),
    "col"       : cols_idx.flatten(),
    "Blue"      : sample[:, :, 0].flatten(),  # BGR order
    "Green"     : sample[:, :, 1].flatten(),
    "Red"       : sample[:, :, 2].flatten(),
    "Gray"      : sample_gray.flatten(),
})

print(f"DataFrame shape: {df.shape}")         # (2500, 6) for 50x50
print(f"\nFirst 8 rows:")
print(df.head(8).to_string(index=False))

print(f"\nStatistical summary (describe):")
print(df[['Blue','Green','Red','Gray']].describe().round(2).to_string())

# Key metrics directly from DataFrame
print(f"\nDerived quality metrics:")
print(f"  Brightness (gray mean) : {df['Gray'].mean():.2f}")
print(f"  Contrast   (gray std)  : {df['Gray'].std():.2f}")
print(f"  Color cast check       : R={df['Red'].mean():.1f}  G={df['Green'].mean():.1f}  B={df['Blue'].mean():.1f}")


# ─────────────────────────────────────────
# SECTION 7: Show images
# ─────────────────────────────────────────

cv2.namedWindow("Original")
cv2.namedWindow("Gray")
cv2.namedWindow("Patch")
cv2.namedWindow("Blue")
cv2.namedWindow("Green")
cv2.namedWindow("Red")

cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Patch", patch)
cv2.imshow("Blue", blue_channel)
cv2.imshow("Green", green_channel)
cv2.imshow("Red", red_channel)

cv2.moveWindow("Original", 0, 0)
cv2.moveWindow("Gray", 650, 0)
cv2.moveWindow("Patch", 1300, 0)

cv2.moveWindow("Blue", 0, 500)
cv2.moveWindow("Green", 650, 500)
cv2.moveWindow("Red", 1300, 500)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("\nDone! All sections completed.")