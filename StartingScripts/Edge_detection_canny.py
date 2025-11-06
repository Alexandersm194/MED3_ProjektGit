import numpy as np
import cv2 as cv


original = cv.imread("TrainingImages//kamelISkygge.jpg")

alpha = 2
beta = -50

kernel = np.ones((15, 15), np.uint8)

contrastBrightness = cv.convertScaleAbs(original, alpha=alpha, beta=beta)
thresholded = cv.threshold(contrastBrightness, 70, 255, cv.THRESH_BINARY)[1]

grayImage = cv.cvtColor(thresholded, cv.COLOR_BGR2GRAY)


def opening(image, inputkernel):
    erosionImg = cv.erode(image, inputkernel, iterations=1)
    dialationImg = cv.dilate(erosionImg, inputkernel, iterations=1)
    return dialationImg

def closing(image, inputkernel):
    dilationImg = cv.dilate(image, inputkernel, iterations=1)
    erosionImg = cv.erode(dilationImg, inputkernel, iterations=1)
    return erosionImg


openImage = opening(grayImage, kernel)
closingImage = closing(grayImage, kernel)

openClose = closing(openImage, kernel)
closeOpen = opening(closingImage, kernel)

thresh = cv.threshold(openClose, 250, 255, cv.THRESH_BINARY)[1]

stuff = cv.Canny(thresh, 100, 225)

cv.imshow("Original", stuff)

cv.waitKey(0)
cv.destroyAllWindows()
