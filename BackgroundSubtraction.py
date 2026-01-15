import cv2 as cv
import numpy as np


def remove_background(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)    #convert to HSV
    H, S, V = cv.split(hsv)                     #split to different channel

    mask_color = cv.inRange(hsv,
                            np.array([0, 150, 0]),  # S > 60
                            np.array([179, 255, 255])) #mask for color - min/max

    mask_black = cv.inRange(hsv,
                            np.array([0, 0, 0]),  # meget lav V
                            np.array([179, 255, 55]))  # undgå almindelige skygger

    # Kombiner masks - for flere farver klodser
    mask_total = cv.bitwise_or(mask_color, mask_black)

    # Rens masken
    kernel = np.ones((5, 5), np.uint8)
    mask_total = cv.morphologyEx(mask_total, cv.MORPH_CLOSE, kernel, iterations=2)  #closes holes, keeps size
    mask_total = cv.morphologyEx(mask_total, cv.MORPH_OPEN, kernel, iterations=2)   #removes noise, keeps size

    # Lav resultatet
    result = cv.bitwise_and(img, img, mask=mask_total)
    return mask_total

