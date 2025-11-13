import numpy as np
import cv2
import matplotlib.pyplot as plt
import BoundingBox
import Matrix
from cv2 import findContours


def rotateImage(orgImg, angle):
    Oheight, Owidth = orgImg.shape[:2]

    # --- Test: rotate the image intentionally ---
    center = (Owidth // 2, Oheight // 2)
    test_rotation = angle
    rotation_matrix = cv2.getRotationMatrix2D(center, test_rotation, 1.0)
    rotated_image = cv2.warpAffine(orgImg, rotation_matrix, (Owidth, Oheight))
    return rotated_image
def opening(image, inputkernel):
    erosionImg = cv2.erode(image, inputkernel, iterations=1)
    dialationImg = cv2.dilate(erosionImg, inputkernel, iterations=1)
    return dialationImg

def closing(image, inputkernel):
    dilationImg = cv2.dilate(image, inputkernel, iterations=1)
    erosionImg = cv2.erode(dilationImg, inputkernel, iterations=1)
    return erosionImg

def blob(input):
    kernel = np.ones((30, 30), np.uint8)
    alpha = 2  # Contrast control (1.0–3.0)
    beta = -50  # Brightness control (0–100)

    # Apply the contrast and brightness adjustment
    adjusted = cv2.convertScaleAbs(input, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    # Define the white range
    # White = low saturation, high brightness
    lower_white = np.array([0, 0, 200])  # H: any, S: low, V: high
    upper_white = np.array([180, 100, 255])  # Allow small variation

    # Create mask for white
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask_inv = cv2.bitwise_not(mask_white)

    # Keep only non-white areas
    backgroundRemoved = cv2.bitwise_and(adjusted, adjusted, mask=mask_inv)

    adjustedBg = cv2.cvtColor(backgroundRemoved, cv2.COLOR_BGR2GRAY)

    output_image = cv2.threshold(adjustedBg, 1, 255, cv2.THRESH_BINARY)[1]

    closedPicture = closing(output_image, kernel)

    stuff = cv2.bitwise_and(input, input, mask=closedPicture)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.imshow("edges", stuff)

    cv2.waitKey(0)
    return closedPicture, stuff

def edge(original):
    alpha = 2
    beta = -50

    kernel = np.ones((15, 15), np.uint8)

    contrastBrightness = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
    thresholded = cv2.threshold(contrastBrightness, 125, 255, cv2.THRESH_BINARY)[1]

    grayImage = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey", grayImage)

    openImage = opening(grayImage, kernel)
    closingImage = closing(grayImage, kernel)

    openClose = closing(openImage, kernel)

    # closeOpen = opening(closingImage, kernel)

    thresh = cv2.threshold(openClose, 250, 255, cv2.THRESH_BINARY)[1]

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

    edge_x = cv2.filter2D(thresh, -1, kernel=sobel_x)
    # edge_x[edge_x != 0] = 255

    edge_y = cv2.filter2D(thresh, -1, kernel=sobel_y)

    edge_x_inv = cv2.filter2D(thresh, -1, kernel=sobel_x_inv)
    # edge_x[edge_x != 0] = 255

    edge_y_inv = cv2.filter2D(thresh, -1, kernel=sobel_y_inv)

    add_edge = edge_x + edge_y + edge_x_inv + edge_y_inv

    return add_edge

def brickEdge(inputImg):
    blur = cv2.blur(inputImg, (23, 23))
    median = cv2.medianBlur(inputImg, 23)

    # adding a gaussian blur
    gaussian_blur = cv2.GaussianBlur(inputImg, (53, 53), 0)
    # adding the two pictures together with larger weight on the original image :D
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
    edges = cv2.Canny(enhanced, 200, 255)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 68, minLineLength=15, maxLineGap=250)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(inputImg, (x1, y1), (x2, y2), (255, 0, 0), 3)


    bluechannel, _ , _ = cv2.split(inputImg)
    lineThresh = cv2.threshold(bluechannel, 250, 255, cv2.THRESH_BINARY)[1]
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.imshow("edges", lineThresh)

    cv2.waitKey(0)
    return lineThresh, edges
def SVD(image):
    height, width = image.shape[:2]

    ys, xs = np.nonzero(image)
    points = np.column_stack((xs, ys))

    # --- Compute dominant direction via covariance + SVD ---
    points_centered = points - points.mean(axis=0)
    cov = np.cov(points_centered, rowvar=False)
    Ut, _, Vt = np.linalg.svd(cov)
    dominant_direction = Vt[0]

    # --- Compute dominant angle (corrected for OpenCV Y-down) ---
    dominant_angle = np.degrees(np.arctan2(-dominant_direction[1], dominant_direction[0]))
    dominant_angle = (dominant_angle + 180) % 180

    print(f"Detected dominant angle: {dominant_angle:.2f}°")

    # --- Compute how much to rotate back upright ---
    correction_angle = 180 - dominant_angle
    print(f"Rotating image by {correction_angle:.2f}° to make upright")

    # --- Rotate the image back upright ---
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    upright_image = cv2.warpAffine(image, M, (width, height))

    # --- Display results ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Before correction ({dominant_angle:.1f}°)")

    plt.subplot(1, 2, 2)
    plt.imshow(upright_image, cmap='gray')
    plt.title("After correction (upright)")

    plt.show()


def dominant_angle_from_lines(image):
    # Canny edges
    edges = cv2.Canny(image, 100, 200)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=20)
    if lines is None:
        print("No lines found.")
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Convert to [0,180) range
    angles = np.mod(angles, 180)

    # Build histogram to find most common angle
    hist, bins = np.histogram(angles, bins=180, range=(0,180))
    dominant_angle = bins[np.argmax(hist)]

    print(f"Detected dominant edge angle: {dominant_angle:.2f}°")

    # Rotate image to make edges vertical (optional)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    correction = dominant_angle - 180
    M = cv2.getRotationMatrix2D(center, correction, 1.0)
    upright = cv2.warpAffine(image, M, (width, height))
    # Show before/after
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Before correction ({dominant_angle:.1f}°)")
    plt.subplot(1,2,2)
    plt.imshow(upright, cmap='gray')
    plt.title("After correction (upright)")
    #plt.show()

    return upright, correction



input_image = cv2.imread("C://Users//Alexa//Documents//GitHub//MED3_ProjektGit//TrainingImages//Fisk.jpg")
test_rotated = rotateImage(input_image, 0)
test_rotate_edgedetected = rotateImage(edge(input_image), 0)
#input_image = rotateImage(brickEdge(input_image)[1], 60)
#input_image = rotateImage(blob(input_image)[1])
angle = dominant_angle_from_lines(test_rotate_edgedetected)[1]

rotated = rotateImage(blob(test_rotated)[0], angle)





cv2.namedWindow("rotated", cv2.WINDOW_NORMAL)
cv2.imshow("rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

crop = BoundingBox.find_bounding_box(rotated)

cv2.namedWindow("rotated", cv2.WINDOW_NORMAL)
cv2.imshow("rotated", crop)

isUp, dotHight, brickHight, brickWidth = Matrix.find_up(crop)
print(isUp)
final_img = crop
if not isUp:
    final_img = rotateImage(crop, 180)

cv2.namedWindow("rotated", cv2.WINDOW_NORMAL)
cv2.imshow("rotated", final_img)
cv2.waitKey(0)

matrix = Matrix.matrix_slice(final_img, brickHight, brickWidth, dotHight)

for y, row in enumerate(matrix):
    for x, col in enumerate(row):
        cv2.imshow(f"{y}, {x}", col)
        cv2.waitKey(0)
#SVD(edge(input_image))
#SVD(input_image)

