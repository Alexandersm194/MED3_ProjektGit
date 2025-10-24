import cv2
import numpy as np
def opening(image, inputkernel):
    erosionImg = cv2.erode(image, inputkernel, iterations=1)
    dialationImg = cv2.dilate(erosionImg, inputkernel, iterations=1)
    return dialationImg

def closing(image, inputkernel):
    dilationImg = cv2.dilate(image, inputkernel, iterations=1)
    erosionImg = cv2.erode(dilationImg, inputkernel, iterations=1)
    return erosionImg


kernel = np.ones((30, 30), np.uint8)


input_image = cv2.imread("C://Users//Alexa//Documents//GitHub//MED3_ProjektGit//TrainingImages//Fisk.jpg")

alpha = 2  # Contrast control (1.0–3.0)
beta = -50     # Brightness control (0–100)

# Apply the contrast and brightness adjustment
adjusted = cv2.convertScaleAbs(input_image, alpha=alpha, beta=beta)

hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

# Define the white range
# White = low saturation, high brightness
lower_white = np.array([0, 0, 200])      # H: any, S: low, V: high
upper_white = np.array([180, 100, 255])   # Allow small variation

# Create mask for white
mask_white = cv2.inRange(hsv, lower_white, upper_white)

mask_inv = cv2.bitwise_not(mask_white)

# Keep only non-white areas
backgroundRemoved = cv2.bitwise_and(adjusted, adjusted, mask=mask_inv)


adjustedBg = cv2.cvtColor(backgroundRemoved, cv2.COLOR_BGR2GRAY)


output_image = cv2.threshold(adjustedBg, 1, 255, cv2.THRESH_BINARY)[1]

closedPicture = closing(output_image, kernel)

subtracted = closedPicture - cv2.erode(closedPicture, kernel, iterations=1)

    #cv2.bitwise_and(input_image, input_image, mask=closedPicture)



cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", closedPicture)
cv2.namedWindow("Fisk", cv2.WINDOW_NORMAL)
cv2.imshow("Fisk", mask_inv)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", input_image)
cv2.namedWindow("Subtracted", cv2.WINDOW_NORMAL)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()
