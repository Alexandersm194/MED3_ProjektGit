import numpy as np
import cv2


def rotateImage(orgImg, angle): #en funktion der roterer ud fra en bestemt vinkel der findes længere nede i koden
    Oheight, Owidth = orgImg.shape[:2]

    # --- Test: rotate the image intentionally ---
    center = (Owidth // 2, Oheight // 2) #find center of image
    test_rotation = angle
    rotation_matrix = cv2.getRotationMatrix2D(center, test_rotation, 1.0)  #finds the rotation matrix
    rotated_image = cv2.warpAffine(orgImg, rotation_matrix, (Owidth, Oheight))  #rotates the images with the rotationmatrix
    return rotated_image
def opening(image, inputkernel): #Basic morphology opening
    erosionImg = cv2.erode(image, inputkernel, iterations=1)
    dialationImg = cv2.dilate(erosionImg, inputkernel, iterations=1)
    return dialationImg

def closing(image, inputkernel): #Basic morphology closing
    dilationImg = cv2.dilate(image, inputkernel, iterations=1)
    erosionImg = cv2.erode(dilationImg, inputkernel, iterations=1)
    return erosionImg


def brickEdge(img): # Her finder vi edges med canny. Der bliver lavet små justeringer der skulle gøre det nemmere at finde edges
    inputImg = img.copy()

    gaussian_blur = cv2.GaussianBlur(inputImg, (53, 53), 0)                 #Gaussian blur
    sharpened_img = cv2.addWeighted(inputImg, 1.7, gaussian_blur, -0.8, 0)  #aplha blend original image and blurred image

    alpha = 3
    beta = 0

    contrastBrightness = cv2.convertScaleAbs(sharpened_img, alpha=alpha, beta=beta) #changing contrast (alpha) and brightness (beta)
    hsv = cv2.cvtColor(contrastBrightness, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, s_thres = cv2.threshold(s, 100, 255, cv2.THRESH_TOZERO) #threshold skulle muligvis gøre det nemmere at finde kanterne. Kan ikke lige huske hvorfor
    _, v_thres = cv2.threshold(v, 50, 255, cv2.THRESH_TOZERO)

    #thresholding on saturation and value, to enchance how clear the edges are (LÆS OP PÅ DET HER)

    hsv_edge_enhanced = cv2.merge([h, s, v_thres])
    enhanced = cv2.cvtColor(hsv_edge_enhanced, cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(enhanced, 200, 255)

# Herfra og ned til lineThresh er det bare for at visualisere
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 68, minLineLength=15, maxLineGap=250)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(inputImg, (x1, y1), (x2, y2), (255, 0, 0), 3)


    bluechannel, _ , _ = cv2.split(inputImg)
    lineThresh = cv2.threshold(bluechannel, 250, 255, cv2.THRESH_BINARY)[1]
# Hertil
    return lineThresh, edges # lineThres er for ren visualisering, edges er det vi har fundet med canny



def dominant_angle_from_lines(img):
    image = img.copy()
    edges = cv2.Canny(image, 100, 200) #Dette er ikke nødvendigt og finder bare lidt flere edges. Kan slettes

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=20) #Den difinerer de dominante linjer, der hvor der er mange linjer efter canny
    if lines is None:
        print("No lines found.")
        return None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    angles = [] # Beregner vinklerne mellem hough og en base linje (vandret) og gemmer dem i et array.
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angles = np.mod(angles, 180) # vi kunne have fundet den dominerende vinkel ud fra den spidse vinkel i stedet??

    hist, bins = np.histogram(angles, bins=180, range=(0,180)) # Vinklerne bliver analyseret i et histogram, så vi kan se hvilke vinkler der optræder mest.
    dominant_angle = bins[np.argmax(hist)] #Gemmer den dominerende vinkel ud fra histogrammet

    print(f"Detected dominant edge angle: {dominant_angle:.2f}°")

    correction = dominant_angle #Gemmer den vinkel som billedet skal korrigeres med

    return correction





