import math

import cv2
import cv2 as cv
import numpy as np
import Segmentation


def removeBorderConnected(img):
    # ensure image is uint8 single channel
    im = img.copy().astype(np.uint8)

    h, w = im.shape[:2]

    # mask MUST be 2 pixels bigger
    mask = np.zeros((h+2, w+2), np.uint8)

    # flood-fill from all border points
    # remove border-connected white (255)
    for x in range(w):
        if im[0, x] == 255:
            cv.floodFill(im, mask, (x, 0), 0)
        if im[h-1, x] == 255:
            cv.floodFill(im, mask, (x, h-1), 0)

    for y in range(h):
        if im[y, 0] == 255:
            cv.floodFill(im, mask, (0, y), 0)
        if im[y, w-1] == 255:
            cv.floodFill(im, mask, (w-1, y), 0)

    return im
def find_up(crop, ref):
    hight, width = ref.shape[:2]
    hightVar = hight // 5
    widthVar = width // 7

    kernel = np.ones((5, 5), np.uint8)
    ref = cv.erode(ref, kernel, iterations=1)
    ref = cv.dilate(ref, kernel, iterations=1)
    corners = [
        ref[0:hightVar, 0:widthVar],
        ref[0:hightVar, (width - widthVar):width],
        ref[(hight - hightVar):hight, 0:widthVar],
        ref[(hight - hightVar):hight, (width - widthVar):width]
    ]



    corrected_corners = []

    for corner in corners:
        corrected_corners.append(removeBorderConnected(corner))

    figure_cnt = None
    biggest_cnt_area = 0
    refCorner = None

    for crn in corrected_corners:
        contours, _ = cv.findContours(crn, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area > biggest_cnt_area:
                figure_cnt = contour
                biggest_cnt_area = area
                refCorner = crn

    cropped, LegoBrickWidth, LegoBrickHeight = Segmentation.find_bounding_box_brick(figure_cnt, refCorner)

    if LegoBrickWidth > LegoBrickHeight:
        LegoBrickWidth, LegoBrickHeight = LegoBrickHeight, LegoBrickWidth

    LegoBrickDotHeight = math.floor(LegoBrickHeight * 0.15)
    print(LegoBrickDotHeight)

    LegoBrickCleanHeight = math.floor(LegoBrickHeight * 0.85)
    print(LegoBrickCleanHeight)
    print(LegoBrickWidth)
    up_brick_kernel = cropped
    down_brick_kernel = cv.rotate(up_brick_kernel, cv.ROTATE_180)
    matchUp = cv.matchTemplate(crop, up_brick_kernel, cv.TM_CCOEFF_NORMED)
    matchDown = cv.matchTemplate(crop, down_brick_kernel, cv.TM_CCOEFF_NORMED)

    # Threshold (float32 result)
    _, matchUp = cv.threshold(matchUp, 0.8, 1.0, cv.THRESH_BINARY)
    _, matchDown = cv.threshold(matchDown, 0.8, 1.0, cv.THRESH_BINARY)

    # Convert to uint8 so findContours works
    matchUp = (matchUp * 255).astype(np.uint8)
    matchDown = (matchDown * 255).astype(np.uint8)


    upCnt, _ = cv.findContours(matchUp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    downCnt, _ = cv.findContours(matchDown, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    isUp = len(upCnt) > len(downCnt)

    isOnSide = len(upCnt) == len(downCnt)


    return isUp, LegoBrickDotHeight, LegoBrickCleanHeight, LegoBrickWidth, isOnSide


def count_bricks_horizontal(img_width, brickWidth, tolerance=0.01):
    bricks = []
    x = 0

    # allowed +- tolerance for measurement error
    min_w = brickWidth * (1 - tolerance)
    max_w = brickWidth * (1 + tolerance)

    while True:
        next_x = x + brickWidth

        # only add the brick if it fully fits within the image (considering max tolerance)
        if next_x <= img_width + max_w:
            bricks.append((x, next_x))
            x = next_x
        else:
            break

    return len(bricks)

def matrix_slice(img, brickHeight, brickWidth, dotHeight=0):
    final_matrix = []
    print(brickWidth, brickHeight)

    img_h, img_w = img.shape[:2]

    # robust detection of horizontal count
    nrBricksHorizontal = count_bricks_horizontal(img_w, brickWidth) - 1
    nrBricksVertical = (img_h - dotHeight) // brickHeight

    print("Detected bricks horizontally:", nrBricksHorizontal)
    print("Detected bricks vertically:", nrBricksVertical)

    '''for y in range(nrBricksVertical):
        row = []
        for x in range(nrBricksHorizontal):
            start_y = y * brickHeight + dotHeight
            end_y   = start_y + brickHeight

            start_x = x * brickWidth
            end_x   = start_x + brickWidth

            # Crop safely (clip if small mismatch)
            start_x = max(0, min(start_x, img_w))
            end_x   = max(0, min(end_x, img_w))

            brickImg = img[start_y:end_y, start_x:end_x]
            row.append(brickImg)

        final_matrix.append(row)'''

    for y in range(nrBricksVertical):
        start_y = y * brickHeight + dotHeight
        end_y = start_y + brickHeight

        brickImg = img[start_y:end_y, 0:img_w]

        brickEdge = cv2.Canny(brickImg, 100, 200)
        final_matrix.append(brickImg)
        cv2.imshow("matrix", brickEdge)
        cv2.waitKey(0)

    return final_matrix



