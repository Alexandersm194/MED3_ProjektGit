import cv2
import numpy as np
import ThresholdTrainer

def classify_brick_size(brick_img, brickHeight, brickWidth):
    brickRatios = []

    for i in range(1, 13):
        if(i % 2 == 0) or i == 1:
            brickRatios.append(brickHeight / (brickWidth * i))
    height, width = brick_img.shape[:2]

    ratio = height / width

    best_match_dis = 10
    best_match = -1

    for index, brick_ratio in enumerate(brickRatios):
        distance = abs(ratio - brick_ratio)
        if distance < best_match_dis:
            best_match_dis = distance
            best_match = index
    if best_match == -1:
        return -1
    else:
        size = best_match + 1
        return size




def classify_brick_hist(hist, trained_histograms, threshold=0.0):

    best_score = -1
    predicted_color = "unknown"

    for color_name, color_hist in trained_histograms.items():
        score = cv2.compareHist(hist.astype(np.float32),
                                color_hist.astype(np.float32),
                                cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            predicted_color = color_name

    # Check if best score is high enough
    if best_score < threshold:
        return "unknown"
    return predicted_color


def classify_brick_mahalanobis(feature, trained_models, threshold=None):
    """
    feature: precomputed 4-element feature vector (cos(h), sin(h), S, V)
    """
    best_color = "unknown"
    best_distance = float('inf')

    for color_name, model in trained_models.items():
        mean_vec = model["mean"]
        inv_cov = model["inv_cov"]

        diff = feature - mean_vec
        d = np.sqrt(float(diff.T @ inv_cov @ diff))
        if d < best_distance:
            best_distance = d
            best_color = color_name

    if threshold is not None and best_distance > threshold:
        return "unknown"

    return best_color
