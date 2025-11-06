import cv2
import numpy as np


def find_line(image):
    cnt, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows, cols = image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt[0], cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    return cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)



def find_box(image):
    cnt, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(cnt)}")
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box


def fit_ellipse(image):
    cnt, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.fitEllipse(cnt[0])

