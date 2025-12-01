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
    up_brick_kernel = cropped[0:LegoBrickDotHeight, 0:LegoBrickWidth]
    down_brick_kernel = cv.rotate(up_brick_kernel, cv.ROTATE_180)

    def get_main_contour(binary_img):
        # Find all contours
        contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Return the largest contour
        return max(contours, key=cv.contourArea)

    def get_contour_angle(contour):
        # Get the minimum-area rectangle around the contour
        rect = cv.minAreaRect(contour)
        angle = rect[-1]
        # Adjust the angle to a range of [-90, 90]
        if angle < -45:
            angle += 90
        return angle

    def is_upright(large_img, kernel_up, kernel_down):
        # Extract main contours
        cnt_large = get_main_contour(large_img)
        cnt_up = get_main_contour(kernel_up)
        cnt_down = get_main_contour(kernel_down)

        if cnt_large is None or cnt_up is None or cnt_down is None:
            print("One of the contours was not found!")
            return None

        # Get angles
        angle_large = get_contour_angle(cnt_large)
        angle_up = get_contour_angle(cnt_up)
        angle_down = get_contour_angle(cnt_down)

        # Compare angles to decide uprightness
        diff_up = abs(angle_large - angle_up)
        diff_down = abs(angle_large - angle_down)

        print(f"Angle large: {angle_large:.2f}, Angle up: {angle_up:.2f}, Angle down: {angle_down:.2f}")
        print(f"Diff up: {diff_up:.2f}, Diff down: {diff_down:.2f}")

        return diff_up < diff_down

    isUp = is_upright(crop, up_brick_kernel, down_brick_kernel)
    print(f"IS UP{isUp}")
    '''top = 0
    bottom = 0


    for y in range(LegoBrickDotHeight):
        for x in range(crop.shape[1]):
            if crop[y, x] == 255:
                top += 1

    for y in range(LegoBrickDotHeight):
        bot = crop.shape[0] - y - 1
        for x in range(crop.shape[1]):
            if crop[bot, x] == 255:
                bottom += 1

    print(f"Top: {top}, Bottom: {bottom}")
    isUp = False if top > bottom else True'''
    return isUp, LegoBrickDotHeight, LegoBrickCleanHeight, LegoBrickWidth


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

    for y in range(nrBricksVertical):
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

        final_matrix.append(row)

    return final_matrix



