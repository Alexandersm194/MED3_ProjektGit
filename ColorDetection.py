import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("TrainingImages/Fisk.jpg")

#///////////////FIRST TRY//////////////////////
'''hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#defining color ranges

upper_green = np.array([149, 174, 94], np.uint8)
lower_green = np.array([84, 114, 40], np.uint8)
green_mask = cv.inRange(hsv, lower_green, upper_green)

upper_yellow = np.array([226, 198, 114], np.uint8)
lower_yellow = np.array([172, 126, 30], np.uint8)
yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

kernal = np.ones((5, 5), "uint8")

green_mask = cv.dilate(green_mask, kernal)
res_green = cv.bitwise_and(img, img, mask = green_mask)

yellow_mask = cv.dilate(green_mask, kernal)
res_yellow = cv.bitwise_and(img, img, mask=green_mask)

contours, hierarcy = cv.findContours(green_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if (area > 200):
        x, y, w, h = cv.boundingRect(contour)
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(img, "Green color", (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


contours, hierarcy = cv.findContours(yellow_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if (area > 200):
        x, y, w, h = cv.boundingRect(contour)
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv.putText(img, "Yellow color", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv.imshow("Color Detection", img)
waitKey(0)
'''
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250])
]
#///////////////SECOND TRY//////////////////////
'''# BGR to HSV
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Defining lower and upper bound HSV values
upper_range = np.array([100, 150, 0])
lower_range = np.array([140, 255, 255])

# Defining mask
mask = cv.inRange(hsv_img, lower_range, upper_range)

result = cv.bitwise_and(img, img, mask=mask)

cv.namedWindow("tihi", cv.WINDOW_NORMAL)
cv.imshow("tihi",img)       # Display original image

cv.namedWindow("masken", cv.WINDOW_NORMAL)
cv.imshow("masken", mask)   # Display blue mask
cv.namedWindow("res", cv.WINDOW_NORMAL)
cv.imshow("res", result)

waitKey(0)'''

alpha = 2
beta = -50

contrastBrightness = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
thresholded = cv.threshold(contrastBrightness, 125, 255, cv.THRESH_BINARY)[1]

# loop over the boundaries
for (lower, upper) in boundaries:

	lower = np.array(lower, dtype = "uint8") #NumPy array from the boundaries
	upper = np.array(upper, dtype = "uint8")

	mask = cv.inRange(img, lower, upper) # find colors within boundaries and apply mask
	output = cv.bitwise_and(thresholded, thresholded, mask = mask)

	cv.namedWindow("images", cv.WINDOW_NORMAL)
	cv.imshow("images", np.hstack([thresholded, output]))
	cv.waitKey(0)


'''cv.imshow("image", thresholded)
waitKey(0)
cv.destroyAllWindows()'''