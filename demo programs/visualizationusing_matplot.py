import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

path = r"C:\RED's space\Wallpapers\71100b488e38a538b67600e93783ba93.jpg"

img = cv2.imread(path)

if img is None:
    print("ERROR: Image not found.")
    exit()

img = cv2.resize(img, (1280, 720))

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

brightness = gray.mean()
contrast = gray.std()
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

print("IMAGE METRICS")
print("Resolution :", img.shape[1], "x", img.shape[0])
print("Brightness :", round(brightness, 2))
print("Contrast   :", round(contrast, 2))
print("Sharpness  :", round(sharpness, 2))

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.hist(gray.flatten(), bins=256, range=[0, 256], color="gray")
plt.title("Brightness Histogram")
plt.xlabel("Brightness")
plt.ylabel("Pixel Count")

plt.subplot(2, 2, 4)

for i, c in enumerate(["blue", "green", "red"]):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=c)

plt.title("RGB Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show(block=True)