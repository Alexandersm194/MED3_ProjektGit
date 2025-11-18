'''coordinates cropout'''
import cv2

# Load the image
img = cv2.imread("C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish1/fish1_12.jpg")

def cropimage(img):
    img_resize = cv2.resize(img, None, fx=0.25, fy=0.25)

    print(img_resize.shape)

    # Define the region of interest (ROI) - arbitrary coordinates
    x_start, y_start, x_end, y_end = 160, 160, 850, 650  # Adjust as needed

    # Crop the image using slicing
    cropped_img = img[y_start:y_end, x_start:x_end]

    print(cropped_img.shape)

    return cropped_img