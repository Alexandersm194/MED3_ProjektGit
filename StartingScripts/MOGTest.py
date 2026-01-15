import cv2
import numpy as np

def RemoveBackgroundStatic(background_images, frame):
    # Compute the median of all background images
    median_bg = np.median(np.array(background_images), axis=0).astype(np.uint8)

    # Ensure frame size matches
    if frame.shape != median_bg.shape:
        frame = cv2.resize(frame, (median_bg.shape[1], median_bg.shape[0]))

    # Compute absolute difference
    diff = cv2.absdiff(frame, median_bg)

    # Convert to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to get mask (tune threshold)
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def RemoveBackground(background, frame):

    # Ensure same size
    if background.shape != frame.shape:
        frame = cv2.resize(frame, (background.shape[1], background.shape[0]))

    # Convert to HSV
    bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    fr_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Compare only H + S channels (ignore V = brightness/shadows)
    diff = cv2.absdiff(fr_hsv[:, :, :2], bg_hsv[:, :, :2])

    # Convert 2-channel diff → 1-channel grayscale by averaging
    diff_gray = diff.mean(axis=2).astype("uint8")

    # Threshold to get foreground mask
    _, mask = cv2.threshold(diff_gray, 80, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


back = cv2.imread("../Backgrounds/Squares.JPG")
image = cv2.imread("../TrainingImages/BrickOnBackground/BrickOnSquares.jpg")

# Generate realistic background variations
def jitter(img):
    noise = np.random.randint(-5, 6, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

backgroundImages = [jitter(back) for _ in range(10)]

#result = RemoveBackgroundStatic(backgroundImages, image)
result = RemoveBackground(back, image)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()