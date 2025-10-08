import numpy as np
import cv2 as cv


original = cv.imread("TrainingImages//Fisk.jpg", cv.IMREAD_GRAYSCALE)

#sobelx = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=3)
#sobely = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=3)
#laplacian = cv.Laplacian(original,cv.CV_64F)

#cv.imshow("Sobel x", sobelx)
#cv.imshow("Sobel y", sobely)
#cv.imshow("Laplacian", laplacian)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

edge_x = cv.filter2D(original, -1, kernel=sobel_x)
edge_y = cv.filter2D(original, -1, kernel=sobel_y)

thresholded = cv.threshold(edge_x, 85, 255, cv.THRESH_BINARY)[1]


cv.imshow ("original picture", edge_x)
cv.imshow ("edge picture", edge_y)

cv.imshow ("threshold", thresholded)

cv.waitKey(0)