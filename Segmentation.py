import cv2
import numpy as np
import math
def find_bounding_box(input):
    contours, _ = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = input.copy()

    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(f"Contours x={x}, y={y}, w={w}, h={h}")

    crop = result[y:y + h, x:x + w]

    return crop, x, y, w, h



