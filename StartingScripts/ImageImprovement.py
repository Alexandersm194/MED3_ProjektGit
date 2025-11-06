import cv2
import cv2 as cv
import numpy as np

img = cv.imread("fishImages/fish2/fish2_12.jpg")

##alternative blur methods:
blur = cv.blur(img,(23,23))
median = cv.medianBlur(img,23)


#adding a gaussian blur
gaussian_blur = cv.GaussianBlur(img, (53, 53), 0)
#adding the two pictures together with larger weight on the original image :D
sharpened_img = cv.addWeighted(img, 1.7, gaussian_blur, -0.8, 0)


cv.namedWindow("original", cv.WINDOW_NORMAL)
cv.imshow("original", img)

cv.namedWindow("gaussian", cv.WINDOW_NORMAL)
cv.imshow("gaussian", gaussian_blur)

cv.namedWindow("sharpened", cv.WINDOW_NORMAL)
cv.imshow("sharpened", sharpened_img)
cv.waitKey(0)


