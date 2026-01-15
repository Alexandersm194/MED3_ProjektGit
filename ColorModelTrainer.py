import os
import cv2
import numpy as np
from glob import glob
from DominantColors import DominantColorsFun


dataset = "color_datasets"


def cluster_to_feature(cluster):
    cluster = np.array(cluster, dtype=np.uint8).flatten()
    if cluster.size != 3:
        raise ValueError(f"Expected BGR with 3 channels, got {cluster.size}")

    pixel = cluster.reshape(1,1,3)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

    H, S, V = hsv
    h_rad = np.deg2rad(H*2.0)

    return np.array([np.cos(h_rad), np.sin(h_rad), S/255.0, V/255.0], dtype=np.float32)

def train_color_mahalanobis():
    color_models = {}
    for color_name in os.listdir(dataset):
        color_path = os.path.join(dataset, color_name)
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
        cov += np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)

        color_models[color_name] = {"mean": mean_vec, "inv_cov": inv_cov}

    return color_models

