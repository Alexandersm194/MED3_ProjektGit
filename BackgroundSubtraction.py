import cv2 as cv
import numpy as np


def remove_background(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)

    mask_color = cv.inRange(hsv,
                            np.array([0, 150, 0]),  # S > 60
                            np.array([179, 255, 255]))

    mask_black = cv.inRange(hsv,
                            np.array([0, 0, 0]),  # meget lav V
                            np.array([179, 255, 55]))  # undgå almindelige skygger

    # Kombiner alle klodser
    mask_total = cv.bitwise_or(mask_color, mask_black)
    # mask_total = cv.bitwise_or(mask_total, mask_white)

    # Rens masken
    kernel = np.ones((5, 5), np.uint8)
    mask_total = cv.morphologyEx(mask_total, cv.MORPH_CLOSE, kernel, iterations=2)
    mask_total = cv.morphologyEx(mask_total, cv.MORPH_OPEN, kernel, iterations=2)

    # Lav resultatet
    result = cv.bitwise_and(img, img, mask=mask_total)
    return mask_total

