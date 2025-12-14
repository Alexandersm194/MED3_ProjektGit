import os
import cv2
import numpy as np
from glob import glob
from DominantColors import DominantColorsFun


dataset_dir = "brickcolor_dataset"


# ---------------------------------------------------------
# HUE circular conversion
# ---------------------------------------------------------

def hue_to_radians(h):
    return np.deg2rad(h * 2.0)  # OpenCV hue 0–180 → 0–360° → radians


# ---------------------------------------------------------
# Convert 1 RGB cluster into HSV + circular hue feature vector
# ---------------------------------------------------------
def cluster_to_feature(cluster):
    """
    Converts BGR to HSV feature: [cos(hue), sin(hue), S_norm, V_norm]
    """
    cluster = np.array(cluster, dtype=np.uint8).flatten()
    if cluster.size != 3:
        raise ValueError(f"Expected BGR with 3 channels, got {cluster.size}")

    pixel = cluster.reshape(1,1,3)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

    H, S, V = hsv
    h_rad = np.deg2rad(H*2.0)  # OpenCV H 0-180 → 0-360 degrees → radians

    return np.array([np.cos(h_rad), np.sin(h_rad), S/255.0, V/255.0], dtype=np.float32)

# ---------------------------------------------------------
# Train Mahalanobis model
# ---------------------------------------------------------
def train_color_mahalanobis():
    color_models = {}
    for color_name in os.listdir(dataset_dir):
        color_path = os.path.join(dataset_dir, color_name)
        if not os.path.isdir(color_path):
            continue

        features = []
        for img_file in glob(os.path.join(color_path, "*.jpg")) + \
                         glob(os.path.join(color_path, "*.png")) + \
                         glob(os.path.join(color_path, "*.jpeg")):
            img = cv2.imread(img_file)
            if img is None:
                continue

            cluster = DominantColorsFun(img)
            feat = cluster_to_feature(cluster)
            features.append(feat)

        if len(features) < 2:
            continue

        X = np.vstack(features)
        mean_vec = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(X.shape[1]) * 1e-6  # regularization
        inv_cov = np.linalg.inv(cov)

        color_models[color_name] = {"mean": mean_vec, "inv_cov": inv_cov}

    return color_models

# ---------------------------------------------------------
# Mahalanobis classifier
# ---------------------------------------------------------
