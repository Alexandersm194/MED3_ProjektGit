import cv2
import numpy as np

inputImg = cv2.imread("TrainingImages/Fisk.jpg")
cv2.imshow("org", inputImg)

hsv = cv2.cvtColor(inputImg, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Unpack threshold result properly
_, v_thres = cv2.threshold(v, 10, 255, cv2.THRESH_TOZERO)

# Merge back
hsv_edge_enhanced = cv2.merge([h, s, v_thres])
enhanced = cv2.cvtColor(hsv_edge_enhanced, cv2.COLOR_HSV2BGR)

cv2.imshow("pic", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()