import cv2
import cv2 as cv
import numpy as np

def find_up(crop, kernel):
    if crop is None or crop.size == 0:
        print("[ERROR] crop image is empty")
        return False, 0, 0, 0, False


    kh, kw = kernel.shape[:2]
    ch, cw = crop.shape[:2]

    if kh >= ch or kw >= cw:
        print("[ERROR] Template (brick) is larger than target image → skip matching.")
        return False, False

    up_kernel = kernel
    down_kernel = cv.rotate(up_kernel, cv.ROTATE_180)

    try:
        matchUp = cv.matchTemplate(crop, up_kernel, cv.TM_CCOEFF_NORMED)
        matchDown = cv.matchTemplate(crop, down_kernel, cv.TM_CCOEFF_NORMED)
    except Exception as e:
        print(f"[ERROR] Template matching failed: {e}")
        return False, False

    # Threshold
    _, matchUp = cv.threshold(matchUp, 0.8, 1.0, cv.THRESH_BINARY)
    _, matchDown = cv.threshold(matchDown, 0.8, 1.0, cv.THRESH_BINARY)

    matchUp = (matchUp * 255).astype(np.uint8)
    matchDown = (matchDown * 255).astype(np.uint8)

    upCnt, _ = cv.findContours(matchUp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    downCnt, _ = cv.findContours(matchDown, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    isUp = len(upCnt) > len(downCnt)
    isSide = len(upCnt) == len(downCnt)

    return isUp, isSide


def count_bricks_horizontal(img_width, brickWidth, tolerance=0.01):
    bricks = []
    x = 0

    max_w = brickWidth * (1 + tolerance)

    while True:
        next_x = x + brickWidth

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

    nrBricksHorizontal = count_bricks_horizontal(img_w, brickWidth) - 1
    nrBricksVertical = (img_h - dotHeight) // brickHeight

    print("Detected bricks horizontally:", nrBricksHorizontal)
    print("Detected bricks vertically:", nrBricksVertical)


    for y in range(nrBricksVertical):
        start_y = y * brickHeight + dotHeight
        end_y = start_y + brickHeight

        brickImg = img[start_y:end_y, 0:img_w]

        brickEdge = cv2.Canny(brickImg, 100, 200)
        final_matrix.append(brickImg)
        cv2.imshow("matrix", brickEdge)
        cv2.waitKey(0)

    return final_matrix



