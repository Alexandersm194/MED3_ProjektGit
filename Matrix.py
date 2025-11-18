import math

import cv2
import cv2 as cv
import numpy as np


def find_up(crop, ref):
    #ref = cv.imread("back.JPG")

    LegoBrickDotHeight = 0
    LegoBrickDotWidth = 0
    LegoBrickCleanHeight = 0
    LegoBrickCleanWidth = 0
    LegoBrickHeight = 0
    LegoBrickWidth = 0

    hight, width = ref.shape[:2]
    hightVar = hight // 2
    widthVar = width // 5

    kernel = np.ones((50, 50), np.uint8)
    corners = [
        ref[0:hightVar, 0:widthVar],
        ref[0:hightVar, (width - widthVar):width],
        ref[(hight - hightVar):hight, 0:widthVar],
        ref[(hight - hightVar):hight, (width - widthVar):width]
    ]

    for corner in corners:
        cv2.imshow("Corner", corner)
        cv2.waitKey(0)



    '''ORANGE_MIN = np.array([5, 50, 50], np.uint8)
    ORANGE_MAX = np.array([15, 255, 255], np.uint8)

    # Process first corner (top-left)
    corner = corners[0]
    hsv_img = cv.cvtColor(corner, cv.COLOR_BGR2HSV)
    frame_threshed = cv.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    closed = cv.morphologyEx(frame_threshed, cv.MORPH_OPEN, kernel, iterations=1)'''

    corner = corners[0]

    contours, _ = cv.findContours(corner, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    figure_cnt = None
    biggest_cnt_area = 0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > biggest_cnt_area:
            figure_cnt = contour
            biggest_cnt_area = area

    if contours:
        x, y, w, h = cv.boundingRect(figure_cnt)
        LegoBrickWidth = w
        LegoBrickHeight = h
        # Since this is top-left corner, no offset needed
        #cv.rectangle(corners[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("No orange object found.")

    LegoBrickDotHeight = math.floor(LegoBrickHeight * 0.15)
    print(LegoBrickDotHeight)

    LegoBrickCleanHeight = math.floor(LegoBrickHeight * 0.85)
    print(LegoBrickCleanHeight)
    print(LegoBrickWidth)

    top = 0
    bottom = 0

    for y in range(LegoBrickDotHeight):
        for x in range(crop.shape[1]):
            print(f"Top: {crop[y, x]}")
            if crop[y, x] == 255:
                top += 1

    for y in range(LegoBrickDotHeight):
        bot = crop.shape[0] - y - 1
        for x in range(crop.shape[1]):
            print(f"Bottom: {crop[bot, x]}")
            if crop[bot, x] == 255:
                bottom += 1

    print(f"Top: {top}, Bottom: {bottom}")
    isUp = False if top > bottom else True
    return isUp, LegoBrickDotHeight, LegoBrickCleanHeight, LegoBrickWidth

def matrix_slice(img, brickHeight, brickWidth, dotHeight=0):
    final_matrix = []

    nrBricksHorizontal = img.shape[1] // brickWidth
    nrBricksVertical = (img.shape[0] - dotHeight) // brickHeight
    print(f"nr of bricksVer: {nrBricksVertical}, nr of bricksHor: {nrBricksHorizontal}")

    for y in range(nrBricksVertical):
        row = []
        for x in range(nrBricksHorizontal):
            start_y = y * brickHeight
            end_y = (y + 1) * brickHeight
            start_x = x * brickWidth
            end_x = (x + 1) * brickWidth

            # Crop brick, skipping top dot region if specified
            brickImg = img[start_y + dotHeight:end_y + dotHeight, start_x:end_x]
            cv2.imshow("Brick", brickImg)
            cv2.waitKey(0)

            row.append(brickImg)

        final_matrix.append(row)


    return final_matrix



