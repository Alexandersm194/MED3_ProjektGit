import cv2 as cv
import numpy as np

def DominantColorsFun(img, clusterAmount=3, crop=0.4):
    h, w, _ = img.shape
    img_crop = img[int(h*crop):int(h*(1-crop)), int(w*crop):int(w*(1-crop))]
    data = img_crop.reshape(-1,3).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0)
    _, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=clusterAmount)
    dominant_idx = int(np.argmax(counts))

    return np.array(centers[dominant_idx], dtype=np.uint8)

