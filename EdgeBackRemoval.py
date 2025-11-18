import cv2 as cv
import numpy as np

original = cv.imread("TrainingImages//Fisk.jpg")

alpha = 2
beta = -50

kernel = np.ones((15, 15), np.uint8)

contrastBrightness = cv.convertScaleAbs(original, alpha=alpha, beta=beta)
thresholded = cv.threshold(contrastBrightness, 125, 255, cv.THRESH_BINARY)[1]

grayImage = cv.cvtColor(thresholded, cv.COLOR_BGR2GRAY)
cv.namedWindow("grey", cv.WINDOW_NORMAL)
cv.imshow("grey", grayImage)
#sobelx = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=3)
#sobely = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=3)
#laplacian = cv.Laplacian(original,cv.CV_64F)

#cv.imshow("Sobel x", sobelx)
#cv.imshow("Sobel y", sobely)
#cv.imshow("Laplacian", laplacian)

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

#closeOpen = opening(closingImage, kernel)

thresh = cv.threshold(openClose, 250, 255, cv.THRESH_BINARY)[1]

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_x_inv = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
sobel_y_inv = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

edge_x = cv.filter2D(thresh, -1, kernel=sobel_x)
#edge_x[edge_x != 0] = 255

edge_y = cv.filter2D(thresh, -1, kernel=sobel_y)

edge_x_inv = cv.filter2D(thresh, -1, kernel=sobel_x_inv)
#edge_x[edge_x != 0] = 255

edge_y_inv = cv.filter2D(thresh, -1, kernel=sobel_y_inv)

add_edge = edge_x + edge_y + edge_x_inv + edge_y_inv

# Threshold combined Sobel edges
_, add_edge_bin = cv.threshold(add_edge, 50, 255, cv.THRESH_BINARY)

# Create empty binary mask
binary_img = np.zeros_like(add_edge_bin)

# Fill horizontally between edges
for y, row in enumerate(add_edge_bin):
    edges_x = np.where(row == 255)[0]
    if len(edges_x) >= 2:
        binary_img[y, edges_x[0]:edges_x[-1]+1] = 255

cv.namedWindow("Filled Mask", cv.WINDOW_NORMAL)
cv.imshow("Filled Mask", binary_img)
cv.waitKey(0)
cv.destroyAllWindows()

