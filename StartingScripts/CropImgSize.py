'''coordinates cropout'''
import cv2

# Load the image
img_path = cv2.imread("C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish1/fish1_12.jpg")

img = cv2.resize(img_path, None, fx=0.25, fy=0.25)

print(img.shape)

# Define the region of interest (ROI) - arbitrary coordinates
x_start, y_start, x_end, y_end = 160, 160, 850, 650  # Adjust as needed

# Crop the image using slicing
cropped_img = img[y_start:y_end, x_start:x_end]

print(cropped_img.shape)

# Show the original and cropped images
cv2.imshow("Original Image", img)
cv2.imshow("Cropped Image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''ROI cropout'''
# import cv2
#
# img_path = cv2.imread("C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish1/fish1_15.jpg")
#
# img = cv2.resize(img_path, None, fx=0.25, fy=0.25)
#
# # Let user select ROI (drag a box)
# roi = cv2.selectROI("Select ROI", img, False)
#
# print(roi)
#
# # Extract cropped region
# cropped_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
#
# # Save and display cropped image
# cv2.imwrite("Cropped.png", cropped_img)
# cv2.imshow("Cropped Image", cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()