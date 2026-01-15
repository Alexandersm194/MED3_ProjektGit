import cv2 as cv
import numpy as np
import Segmentation
def removeBorderConnected(img):
    im = img.copy().astype(np.uint8)

    h, w = im.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

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
def findReferenceBrick(ref):
    if ref is None or ref.size == 0:
        print("[ERROR] reference image is empty")
        return False, 0, 0, 0, False

    h, w = ref.shape[:2]
    hVar = max(1, h // 5)
    wVar = max(1, w // 7)

    kernel = np.ones((5, 5), np.uint8)
    ref_clean = cv.erode(ref, kernel, iterations=1)
    ref_clean = cv.dilate(ref_clean, kernel, iterations=1)

    corners = [
        ref_clean[0:hVar, 0:wVar],
        ref_clean[0:hVar, w - wVar:w],
        ref_clean[h - hVar:h, 0:wVar],
        ref_clean[h - hVar:h, w - wVar:w]
    ]

    corrected_corners = [removeBorderConnected(c) for c in corners]

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

    if figure_cnt is None or len(figure_cnt) == 0:
        print("[ERROR] No corner contour found → cannot determine orientation.")
        return None, 0, 0, 0

    try:
        cropped, bw, bh = Segmentation.find_bounding_box_brick(figure_cnt, refCorner)
    except Exception as e:
        print(f"[ERROR] find_bounding_box_brick failed: {e}")
        return None, 0, 0, 0

    if cropped is None or cropped.size == 0:
        print("[ERROR] Cropped brick is empty.")
        return None, 0, 0, 0

    if bw > bh:
        bw, bh = bh, bw

    dotHeight = int(bh * 0.15)
    cleanHeight = int(bh * 0.85)

    return cropped, dotHeight, cleanHeight, bw