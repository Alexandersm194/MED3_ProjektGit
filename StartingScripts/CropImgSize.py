'''coordinates cropout'''
import cv2

# Load the image
img_path = cv2.imread("C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish2/fish2_1.jpg")

img = cv2.resize(img_path, None, fx=0.25, fy=0.25)

print(img.shape)

# Define the region of interest (ROI) - arbitrary coordinates
x_start, y_start, x_end, y_end = 160, 160, 850, 650  # Adjust as needed

# Crop the image using slicing

cropped_img = img[y_start:y_end, x_start:x_end]

print(cropped_img.shape)

cv2.imshow("image", img)
cv2.imshow("cropped image", cropped_img)
cv2.waitKey(0)