import cv2 as cv
import numpy as np
import math
def crop_rotated_rect(image, rect):
    # Get box corner points
    box = cv.boxPoints(rect)
    box = np.int8(box)

    # Width and height of the rect
    W = int(rect[1][0])
    H = int(rect[1][1])

    # Destination points for the perspective transform
    dst_pts = np.array([
        [0, H-1],
        [0, 0],
        [W-1, 0],
        [W-1, H-1]
    ], dtype="float32")

    # Order box points consistently
    src_pts = box.astype("float32")

    # Get the perspective transform matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Warp (crop) the image
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

