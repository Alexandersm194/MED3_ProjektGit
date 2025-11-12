import cv2 as cv
import numpy as np

img = cv.imread("back.JPG")

hight, width = img.shape[:2]
hightVar = hight // 4
widthVar = width // 5

kernel = np.ones((50, 50), np.uint8)
corners = [
    img[0:hightVar, 0:widthVar],
    img[0:hightVar, (width - widthVar):width],
    img[(hight - hightVar):hight, 0:widthVar],
    img[(hight - hightVar):hight, (width - widthVar):width]
]

ORANGE_MIN = np.array([5, 50, 50], np.uint8)
ORANGE_MAX = np.array([15, 255, 255], np.uint8)

# Process first corner (top-left)
corner = corners[0]
hsv_img = cv.cvtColor(corner, cv.COLOR_BGR2HSV)
frame_threshed = cv.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
closed = cv.morphologyEx(frame_threshed, cv.MORPH_OPEN, kernel, iterations=1)

contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if contours:
    x, y, w, h = cv.boundingRect(contours[0])
    print(w, h)
    # Since this is top-left corner, no offset needed
    cv.rectangle(corners[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("No orange object found.")

cv.imshow("mask", closed)
cv.imshow("image with box", corners[0])
cv.waitKey(0)
cv.destroyAllWindows()
