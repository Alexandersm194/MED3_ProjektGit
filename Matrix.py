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
    """
    Detect brick orientation + compute dot height + clean height + width.
    Guaranteed safe: no OpenCV crashes, no empty contours, no invalid kernels.
    """

    # ---------------------------
    # 0. Validate input images
    # ---------------------------
    if crop is None or crop.size == 0:
        print("[ERROR] crop image is empty")
        return False, 0, 0, 0, False

    if ref is None or ref.size == 0:
        print("[ERROR] reference image is empty")
        return False, 0, 0, 0, False

    # ---------------------------
    # 1. Corner extraction
    # ---------------------------
    h, w = ref.shape[:2]
    hVar = max(1, h // 5)
    wVar = max(1, w // 7)

    kernel = np.ones((5, 5), np.uint8)
    ref_clean = cv.erode(ref, kernel, iterations=1)
    ref_clean = cv.dilate(ref_clean, kernel, iterations=1)

    # Extract 4 corners safely
    corners = [
        ref_clean[0:hVar, 0:wVar],
        ref_clean[0:hVar, w - wVar:w],
        ref_clean[h - hVar:h, 0:wVar],
        ref_clean[h - hVar:h, w - wVar:w]
    ]

    corrected_corners = [removeBorderConnected(c) for c in corners]

    # ---------------------------
    # 2. Find biggest contour in all corners
    # ---------------------------
    figure_cnt = None
    biggest_area = 0
    refCorner = None

    for crn in corrected_corners:
        if crn is None or crn.size == 0:
            continue

        cnts, _ = cv.findContours(crn, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv.contourArea(c)
            if area > biggest_area:
                biggest_area = area
                figure_cnt = c
                refCorner = crn

    # ---------------------------
    # 3. VALIDATION: did we find a contour?
    # ---------------------------
    if figure_cnt is None or len(figure_cnt) == 0:
        print("[ERROR] No corner contour found → cannot determine orientation.")
        return False, 0, 0, 0, False

    # ---------------------------
    # 4. Compute bounding box brick
    # ---------------------------
    try:
        cropped, bw, bh = Segmentation.find_bounding_box_brick(figure_cnt, refCorner)
    except Exception as e:
        print(f"[ERROR] find_bounding_box_brick failed: {e}")
        return False, 0, 0, 0, False

    # Ensure valid bounding-box brick
    if cropped is None or cropped.size == 0:
        print("[ERROR] Cropped brick is empty.")
        return False, 0, 0, 0, False

    # Normalize width/height
    if bw > bh:
        bw, bh = bh, bw

    dotHeight = int(bh * 0.15)
    cleanHeight = int(bh * 0.85)

    # ---------------------------
    # 5. TEMPLATE MATCHING SAFETY CHECK
    # ---------------------------
    kh, kw = cropped.shape[:2]
    ch, cw = crop.shape[:2]

    if kh >= ch or kw >= cw:
        print("[ERROR] Template (brick) is larger than target image → skip matching.")
        return False, dotHeight, cleanHeight, bw, False

    # Kernel for matching
    up_kernel = cropped
    down_kernel = cv.rotate(up_kernel, cv.ROTATE_180)

    # ---------------------------
    # 6. Template matching
    # ---------------------------
    try:
        matchUp = cv.matchTemplate(crop, up_kernel, cv.TM_CCOEFF_NORMED)
        matchDown = cv.matchTemplate(crop, down_kernel, cv.TM_CCOEFF_NORMED)
    except Exception as e:
        print(f"[ERROR] Template matching failed: {e}")
        return False, dotHeight, cleanHeight, bw, False

    # Threshold
    _, matchUp = cv.threshold(matchUp, 0.8, 1.0, cv.THRESH_BINARY)
    _, matchDown = cv.threshold(matchDown, 0.8, 1.0, cv.THRESH_BINARY)

    matchUp = (matchUp * 255).astype(np.uint8)
    matchDown = (matchDown * 255).astype(np.uint8)

    upCnt, _ = cv.findContours(matchUp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    downCnt, _ = cv.findContours(matchDown, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    isUp = len(upCnt) > len(downCnt)
    isSide = len(upCnt) == len(downCnt)

    return isUp, dotHeight, cleanHeight, bw, isSide


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



