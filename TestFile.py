import cv2

image = cv2.imread('TrainingImages//Klods.jpg')


cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

