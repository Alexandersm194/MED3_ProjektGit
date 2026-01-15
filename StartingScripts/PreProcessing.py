import cv2
import numpy as np


def remove_shadows_preserve_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Large blur to estimate shadow
    l_blur = cv2.GaussianBlur(l, (101, 101), 0)

    # Normalize L
    l_corrected = cv2.divide(l, l_blur, scale=255)

    # Merge with original color
    lab_corrected = cv2.merge([l_corrected, a, b])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return result

def sharpened_image(image):
    blur_amount = 23
    gauss_amount = 23

    blur = cv2.blur(image, (blur_amount, blur_amount))
    gauss = cv2.GaussianBlur(blur, (gauss_amount, gauss_amount), 0)

    sharpened_img = cv2.addWeighted(image, 1.7, blur, -0.8, 0)

    return sharpened_img
