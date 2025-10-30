import cv2
import numpy as np

inputImg = cv2.imread("TrainingImages/Fisk.jpg")
cv2.imshow("org", inputImg)

blur = cv2.blur(inputImg,(23, 23))
median = cv2.medianBlur(inputImg, 23)


#adding a gaussian blur
gaussian_blur = cv2.GaussianBlur(inputImg, (53, 53), 0)
#adding the two pictures together with larger weight on the original image :D
sharpened_img = cv2.addWeighted(inputImg, 1.7, gaussian_blur, -0.8, 0)

alpha = 3
beta = 0

kernel = np.ones((15, 15), np.uint8)

contrastBrightness = cv2.convertScaleAbs(sharpened_img, alpha=alpha, beta=beta)
hsv = cv2.cvtColor(contrastBrightness, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Unpack threshold result properly
_, s_thres = cv2.threshold(s, 100, 255, cv2.THRESH_TOZERO)
_, v_thres = cv2.threshold(v, 50, 255, cv2.THRESH_TOZERO)

# Merge back
hsv_edge_enhanced = cv2.merge([h, s, v_thres])
enhanced = cv2.cvtColor(hsv_edge_enhanced, cv2.COLOR_HSV2BGR)
cv2.imshow("pic", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()