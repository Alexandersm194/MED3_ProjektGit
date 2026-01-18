import os
import cv2
import numpy as np
from glob import glob
from DominantColors import DominantColorsFun


dataset = "color_datasets"

# omdanner clusters til features
def cluster_to_feature(cluster):
    cluster = np.array(cluster, dtype=np.uint8).flatten()
    if cluster.size != 3:
        raise ValueError(f"Expected BGR with 3 channels, got {cluster.size}")

# Konverterer mean BGR værdien (fra clusteren). Til et billede bestående af en enkelt pixel med 3 channels (BGR). Dette gres da cvtcolor kun kan bruges på billeder og ikke værdier.
    pixel = cluster.reshape(1,1,3)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

# vi laver den cirkulær, da vi havde problemer med at rød blev set som blå.
    H, S, V = hsv
    h_rad = np.deg2rad(H*2.0)

# retnerer en feature vektor
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

#Hver farve bliver inddelt i et array som indeholder alle feature vektorer for hver farve.
            cluster = DominantColorsFun(img)
            feat = cluster_to_feature(cluster)
            features.append(feat)

        if len(features) < 2:
            continue

# for hver farve/feature finder der en mean vektor. Invers covarians findes også da mahalanobis skal bruge disse.
        X = np.vstack(features)
        mean_vec = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)

        color_models[color_name] = {"mean": mean_vec, "inv_cov": inv_cov}

    return color_models

