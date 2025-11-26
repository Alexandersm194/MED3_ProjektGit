import numpy as np
import cv2


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

    lower_white = np.array([0, 0, 180])  # Only brighter whites
    upper_white = np.array([180, 50, 255])

    # Create mask for white
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask_inv = cv2.bitwise_not(mask_white)




    # Keep only non-white areas
    backgroundRemoved = cv2.bitwise_and(adjusted, adjusted, mask=mask_inv)

    adjustedBg = cv2.cvtColor(backgroundRemoved, cv2.COLOR_BGR2GRAY)

    output_image = cv2.threshold(adjustedBg, 1, 255, cv2.THRESH_BINARY)[1]

    closedPicture = closing(output_image, kernel)

    stuff = cv2.bitwise_and(input, input, mask=closedPicture)

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

def brickEdge(img):
    inputImg = img.copy()
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



def dominant_angle_from_lines(img):
    image = img.copy()
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

    correction = dominant_angle - 180

    return correction





