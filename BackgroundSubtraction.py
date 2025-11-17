import cv2
import numpy as np
import os

# --- Konfiguration ---
BG_DIR = r"C:\Users\Admin\Documents\GitHub\MED3_ProjektGit\TrainingMaterial\square"  # mappe med baggrundsbilleder uden LEGO
INPUT_IMAGE = r"C:\Users\Admin\Documents\GitHub\MED3_ProjektGit\TrainingImages\BrickOnBackground\BrickOnSquares.jpg"
OUTPUT_DIR = r"C:\Users\Admin\Documents\GitHub\MED3_ProjektGit\output"
THRESHOLD = 40  # farveafvigelse threshold (juster)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Indlæs baggrundsbilleder ---
bg_images = []
for fname in os.listdir(BG_DIR):
    path = os.path.join(BG_DIR, fname)
    img = cv2.imread(path)
    if img is not None:
        bg_images.append(img.astype(np.float32))

if not bg_images:
    raise RuntimeError("Ingen baggrundsbilleder fundet!")

# --- 2. Beregn gennemsnit baggrundsmodel ---
bg_stack = np.stack(bg_images, axis=3)  # HxWx3xN
bg_model = np.mean(bg_stack, axis=3)    # HxWx3

# --- 3. Læs LEGO-billedet ---
img = cv2.imread(INPUT_IMAGE).astype(np.float32)
if img.shape != bg_model.shape:
    raise RuntimeError("Billedstørrelse matcher ikke baggrundsmodellen!")

# --- 4. Beregn masken ---
diff = np.linalg.norm(img - bg_model, axis=2)  # Euclidean distance per pixel
mask = (diff > THRESHOLD).astype(np.uint8)

# --- 5. Rens masken ---
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.medianBlur(mask, 5)

# Connected components for at fjerne små pixels
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
min_size = 100
mask_clean = np.zeros_like(mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        mask_clean[labels == i] = 1
mask = mask_clean

# --- 6. Gem gennemsigtigt billede ---
result = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask * 255
cv2.imwrite(os.path.join(OUTPUT_DIR, "lego_no_bg.png"), result)

print("✅ Færdig! Baggrund fjernet.")