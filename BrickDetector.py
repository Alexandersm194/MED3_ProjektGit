import cv2 as cv
import numpy as np
import Segmentation
def sort_bricks_grid(brick_boxes, brick_images):
    boxes = np.array(brick_boxes)

    # Sort by Y first → rows
    sort_y = np.argsort(boxes[:, 1])
    boxes = boxes[sort_y]
    images = [brick_images[i] for i in sort_y]

    # Group into rows based on similar Y
    rows = []
    current_row = [0]
    threshold = 20  # adjust depending on spacing

    for i in range(1, len(boxes)):
        if abs(boxes[i, 1] - boxes[current_row[0], 1]) < threshold:
            current_row.append(i)
        else:
            rows.append(current_row)
            current_row = [i]
    rows.append(current_row)

    # Now sort each row by X
    final_images = []
    final_boxes = []

    for row in rows:
        row_sorted = sorted(row, key=lambda i: float(boxes[i][0]))

        # store images
        final_images.append([images[i] for i in row_sorted])

        # store boxes (converted back to tuples)
        final_boxes.append([tuple(boxes[i]) for i in row_sorted])

    return final_images, final_boxes
def brick_detect(corrected_img, corrected_img_bin, brickWidth, brickHeight,
                 dotHeight=0, center_bias_x=0.9):

    hsv = cv.cvtColor(corrected_img, cv.COLOR_BGR2LAB)
    h, s, v = cv.split(hsv)

    # Increase contrast by scaling V
    alpha = 2 # contrast factor (>1 = more contrast)
    v = np.clip(v * alpha, 0, 255).astype(np.uint8)

    # Merge back
    hsv_contrast = cv.merge([h, s, v])
    edge = cv.Canny(hsv_contrast, 100, 200)
    edge = cv.dilate(edge, np.ones((5, 5), np.uint8), iterations=2)

    bricks_mask = corrected_img_bin - edge

    bricks_mask = cv.morphologyEx(bricks_mask, cv.MORPH_OPEN,
                                  np.ones((40, 40), np.uint8), iterations=1)


    cnts, _ = cv.findContours(bricks_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    brick_images = []
    brick_boxes = []

    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area < 2000:
            continue

        x, y, w, h = cv.boundingRect(cnt)

        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(corrected_img.shape[1] - x, w + pad * 2)
        h = min(corrected_img.shape[0] - y, h + pad * 2)

        crop = corrected_img[y:y + h, x:x + w].copy()

        brick_images.append(crop)
        brick_boxes.append((x, y, w, h))

        cv.rectangle(corrected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    '''cv.namedWindow("Detected bricks", cv.WINDOW_NORMAL)
    cv.imshow("Detected bricks", corrected_img)
    cv.waitKey(0)'''

    # 3. Sort bricks into rows and then by X
    sorted_images, sorted_boxes = sort_bricks_grid(brick_boxes, brick_images)

    img_h, img_w = corrected_img.shape[:2]
    nrBricksHorizontal = img_w // brickWidth
    nrBricksVertical = (img_h - dotHeight) // brickHeight

    # --- assign images by biased center ---
    final_grid = [[None for _ in range(nrBricksHorizontal)]
                  for _ in range(nrBricksVertical)]

    for row_imgs, row_boxes in zip(sorted_images, sorted_boxes):
        for img, box in zip(row_imgs, row_boxes):
            if box is None:
                continue

            x, y, w, h = box

            # Apply horizontal left-bias
            cx_biased = x + w * (1 - center_bias_x)
            cy_center = y + h / 2

            grid_x = min(int(cx_biased // brickWidth), nrBricksHorizontal - 1)
            grid_y = min(int((cy_center - dotHeight) // brickHeight), nrBricksVertical - 1)

            final_grid[grid_y][grid_x] = img

    return final_grid

