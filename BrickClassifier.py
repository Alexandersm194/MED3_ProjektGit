import cv2
import numpy as np

def classify_brick_hist(hist, trained_histograms, threshold=0.0):
    """
    hist: normalized HSV histogram of one brick
    trained_histograms: dict of color_name -> histogram
    threshold: minimum correlation to accept classification
    returns: predicted color name or 'unknown'
    """
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