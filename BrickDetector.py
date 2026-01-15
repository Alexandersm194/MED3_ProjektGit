import cv2 as cv
import numpy as np
import Segmentation
def sort_bricks_grid(brick_boxes, brick_images):
    boxes = np.array(brick_boxes)

    sort_y = np.argsort(boxes[:, 1])
    boxes = boxes[sort_y]
    images = [brick_images[i] for i in sort_y]

    rows = []
    current_row = [0]
    threshold = 20

    for i in range(1, len(boxes)):
        if abs(boxes[i, 1] - boxes[current_row[0], 1]) < threshold:
            current_row.append(i)
        else:
            rows.append(current_row)
            current_row = [i]
    rows.append(current_row)

    final_images = []
    final_boxes = []

    for row in rows:
        row_sorted = sorted(row, key=lambda i: float(boxes[i][0]))

        final_images.append([images[i] for i in row_sorted])

        final_boxes.append([tuple(boxes[i]) for i in row_sorted])

    return final_images, final_boxes
def brick_detect(corrected_img, corrected_img_bin, brickWidth, brickHeight,
                 dotHeight=0, center_bias_x=0.9):

    if corrected_img is None or corrected_img.size == 0:
        print("[ERROR] corrected_img is empty")
        return []

    if corrected_img_bin is None or corrected_img_bin.size == 0:
        print("[ERROR] corrected_img_bin is empty")
        return []

    if brickWidth <= 0 or brickHeight <= 0:
        print(f"[ERROR] Invalid brick size: width={brickWidth}, height={brickHeight}")
        return []

    img_h, img_w = corrected_img.shape[:2]

    lab = cv.cvtColor(corrected_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    alpha = 2
    l = np.clip(l * alpha, 0, 255).astype(np.uint8)

    edges = cv.Canny(l, 100, 200)
    edges = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    bricks_mask = corrected_img_bin - edges
    bricks_mask = cv.morphologyEx(bricks_mask, cv.MORPH_OPEN,
                                  np.ones((40, 40), np.uint8), iterations=1)

    cnts, _ = cv.findContours(bricks_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] No contours found")
        return []

    brick_images = []
    brick_boxes = []

    # 4. Extract valid bricks
    for cnt in cnts:
        if cv.contourArea(cnt) < 2000:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_w - x, w + pad * 2)
        h = min(img_h - y, h + pad * 2)

        crop = corrected_img[y:y + h, x:x + w].copy()
        if crop.size == 0:
            continue

        brick_images.append(crop)
        brick_boxes.append((x, y, w, h))
        cv.rectangle(corrected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.namedWindow("Detected bricks", cv.WINDOW_NORMAL)
    cv.imshow("Detected bricks", corrected_img)
    cv.waitKey(0)

    if not brick_images:
        print("[WARN] No valid bricks after area filtering")
        return []

    # 5. Sort bricks by row then column
    sorted_images, sorted_boxes = sort_bricks_grid(brick_boxes, brick_images)

    nrBricksHorizontal = max(1, img_w // brickWidth)
    nrBricksVertical = max(1, (img_h - dotHeight) // brickHeight)

    final_grid = [[None for _ in range(nrBricksHorizontal)]
                  for _ in range(nrBricksVertical)]

    for row_imgs, row_boxes in zip(sorted_images, sorted_boxes):
        for img, box in zip(row_imgs, row_boxes):
            if img is None or box is None:
                continue

            x, y, w, h = box

            cx_biased = x + w * (1 - center_bias_x)
            cy_center = y + h / 2

            grid_x = int(cx_biased // brickWidth)
            grid_y = int((cy_center - dotHeight) // brickHeight)

            if grid_x < 0 or grid_x >= nrBricksHorizontal:
                print(f"[SKIP] grid_x out of bounds: {grid_x}")
                continue
            if grid_y < 0 or grid_y >= nrBricksVertical:
                print(f"[SKIP] grid_y out of bounds: {grid_y}")
                continue

            final_grid[grid_y][grid_x] = img

    return final_grid

