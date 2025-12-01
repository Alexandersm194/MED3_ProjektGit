import cv2 as cv
import numpy as np
import math

def background_removal(image):
    # Indlæs billede

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Definér område for farvede klodser
    lower_color = np.array([0, 150, 0])  # Lav mætning og lysstyrke udelukkes
    upper_color = np.array([180, 255, 255])  # Alt med farve beholdes

    # Definér område for mørke klodser
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([179, 255, 55])

    # Lav maske
    mask_color = cv.inRange(hsv, lower_color, upper_color)
    mask_dark = cv.inRange(hsv, lower_dark, upper_dark)

    # Kombinerede masker
    mask = cv.bitwise_or(mask_color, mask_dark)

    kernel = np.ones((5, 5), np.uint8)
    closedPic = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    closedPic = cv.morphologyEx(closedPic, cv.MORPH_OPEN, kernel, iterations=3)
    # Vis resultat
    result = cv.bitwise_and(image, image, mask=closedPic)


    return closedPic, result

def crop_rotated_rect(image, rect):
    # rect = ((cx, cy), (w, h), angle)
    box = cv.boxPoints(rect).astype(np.float32)

    # Order points (TL, TR, BR, BL)
    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]      # top-left
        ordered[2] = pts[np.argmax(s)]      # bottom-right
        ordered[1] = pts[np.argmin(diff)]   # top-right
        ordered[3] = pts[np.argmax(diff)]   # bottom-left
        return ordered

    src = order_points(box)

    # Compute width and height from the actual rotated box
    W = int(np.linalg.norm(src[0] - src[1]))
    H = int(np.linalg.norm(src[0] - src[3]))

    # Destination points that preserve orientation exactly
    dst = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ], dtype="float32")

    # Compute transform and warp
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(image, M, (W, H))

    return warped, W, H

def find_bounding_box(input):
    contours, _ = cv.findContours(input, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = input.copy()

    figure_cnt = None
    biggest_cnt_area = 0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > biggest_cnt_area:
            figure_cnt = contour
            biggest_cnt_area = area

    x, y, w, h = cv.boundingRect(figure_cnt)
    cv.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(f"Contours x={x}, y={y}, w={w}, h={h}")

    crop = result[y:y + h, x:x + w]

    return crop, x, y, w, h



def find_bounding_box_brick(input, org):
    rect = cv.minAreaRect(input)
    corrected, w, h = crop_rotated_rect(org, rect)

    return corrected, w, h

