import cv2 as cv
import numpy as np
import Segmentation
def removeBorderConnected(img): # Denne kode fjerne støj der støder op til borderen. Dette gøres for at fjerne støj. Der opstod ofte støj omkring borderen
    im = img.copy().astype(np.uint8)

    h, w = im.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

    for x in range(w): # floodFill er en blob detection metode hvor vi ændrer værdien.
        if im[0, x] == 255:
            cv.floodFill(im, mask, (x, 0), 0) # seed point er der vi starter (hvor vi starter floodFill). newVal er den nye værdi som blobben skal have
        if im[h-1, x] == 255:
            cv.floodFill(im, mask, (x, h-1), 0)

    for y in range(h):
        if im[y, 0] == 255:
            cv.floodFill(im, mask, (0, y), 0)
        if im[y, w-1] == 255:
            cv.floodFill(im, mask, (w-1, y), 0)

    return im
def findReferenceBrick(ref): # Dette del af koden tager et binært billede
    if ref is None or ref.size == 0:
        print("[ERROR] reference image is empty")
        return False, 0, 0, 0, False

    h, w = ref.shape[:2] # Vi definerer hvor store dele af hjørnet som vi skal fokusere på
    hVar = max(1, h // 5)
    wVar = max(1, w // 7)

    kernel = np.ones((5, 5), np.uint8)  #renserr billdet med morphologi
    ref_clean = cv.erode(ref, kernel, iterations=1)
    ref_clean = cv.dilate(ref_clean, kernel, iterations=1)

# Cropper hvert hjørne, så de bliver hver sit billede
    corners = [
        ref_clean[0:hVar, 0:wVar],
        ref_clean[0:hVar, w - wVar:w],
        ref_clean[h - hVar:h, 0:wVar],
        ref_clean[h - hVar:h, w - wVar:w]
    ]

# Fix noise i alle Corners (se over), så vi får et reference billede uden støj i hjørnerne
    corrected_corners = [removeBorderConnected(c) for c in corners]

    figure_cnt = None
    biggest_area = 0
    refCorner = None

# Denne kode kører igennem alle hjørnerne på papiret og finder contours. Derefter finder den den største contours ud af alle fire hjørner.
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

# Tager den reference som blev fundet i koden over. Den finder en bounding box (og cropper ind), højden og bredden.
    try:
        cropped, bw, bh = Segmentation.find_bounding_box_brick(figure_cnt, refCorner)
    except Exception as e:
        print(f"[ERROR] find_bounding_box_brick failed: {e}")
        return None, 0, 0, 0

    if cropped is None or cropped.size == 0:
        print("[ERROR] Cropped brick is empty.")
        return None, 0, 0, 0

# Sørger for at bredden altid er den korteste
    if bw > bh:
        bw, bh = bh, bw

# vi ved at stud er 15% af højden, så vi finder dens højde på nedstående måde. Her finder vi også højde på klods uden stud
    dotHeight = int(bh * 0.15)
    cleanHeight = int(bh * 0.85)

    return cropped, dotHeight, cleanHeight, bw