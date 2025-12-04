import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
from Segmentation import background_removal, find_bounding_box
from DominantColors import DominantColorsFun  # assumes it returns all cluster centers

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
dataset_dir = "brickcolor_dataset"
OUTPUT_FILE = "lego_color_histograms.npy"

H_BINS = 30
S_BINS = 32
V_BINS = 32

# ---------------------------------------------------------
# Convert cluster colors to HSV histogram
# ---------------------------------------------------------
def clusters_to_hist(clusters):
    """
    clusters: list of BGR tuples from DominantColorsFun
    returns: concatenated HSV histogram
    """
    h_hist = np.zeros(H_BINS, dtype=np.float32)
    s_hist = np.zeros(S_BINS, dtype=np.float32)
    v_hist = np.zeros(V_BINS, dtype=np.float32)

    for bgr in clusters:
        pixel = np.uint8([[bgr]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

        h_bin = int(hsv[0] / 180 * H_BINS)
        s_bin = int(hsv[1] / 256 * S_BINS)
        v_bin = int(hsv[2] / 256 * V_BINS)

        h_bin = min(H_BINS - 1, h_bin)
        s_bin = min(S_BINS - 1, s_bin)
        v_bin = min(V_BINS - 1, v_bin)

        h_hist[h_bin] += 1
        s_hist[s_bin] += 1
        v_hist[v_bin] += 1

    # Normalize
    hist = np.concatenate([h_hist, s_hist, v_hist])
    hist /= hist.sum()
    return hist

# ---------------------------------------------------------
# TRAINING: compute histogram per color using all clusters
# ---------------------------------------------------------
def train_color_histograms():
    color_histograms = {}

    for color_name in os.listdir(dataset_dir):
        color_path = os.path.join(dataset_dir, color_name)
        if not os.path.isdir(color_path):
            continue

        print(f"Processing color: {color_name}")

        all_hist = []

        for img_file in glob(os.path.join(color_path, "*.jpg")) + \
                         glob(os.path.join(color_path, "*.png")) + \
                         glob(os.path.join(color_path, "*.jpeg")):

            img = cv2.imread(img_file)
            if img is None:
                continue
            imgBin = background_removal(img)[0]
            cropped_bin, x, y, w, h = find_bounding_box(imgBin)
            img = img[y:y + h, x:x + w]
            img = background_removal(img)[1]
            cv2.imshow("test", img)
            cv2.waitKey(0)

            # Use DominantColorsFun to get cluster centers (all 3 clusters)
            clusters = DominantColorsFun(img)  # modify to return list of tuples if not already
            if not isinstance(clusters, list):
                clusters = [clusters]

            hist = clusters_to_hist(clusters)
            all_hist.append(hist)

        if all_hist:
            mean_hist = np.mean(all_hist, axis=0)
            color_histograms[color_name] = mean_hist
            print(f"{color_name}: histogram computed")

    return color_histograms

