import cv2 as cv
import numpy as np
import ModelDirection as MD
import os


def DirecetionEval(img):
    yIn = img.shape[0] / 5.4
    xIn = img.shape[1] / 8

    # convert to integers for cropping
    yIn = int(yIn)
    xIn = int(xIn)
    figureImg = img[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]

    edge = MD.brickEdge(figureImg)[1]


    # Direction
    dominant_angle = MD.dominant_angle_from_lines(edge)
    return dominant_angle

programDir = ""
expectedAngle = 0
programImages = []

if os.path.isdir(programDir):
    for file in os.listdir(programDir):
        full_path = os.path.join(programDir, file)
        img = cv.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            programImages.append(img)

gatheredError = 0

for image in programImages:
    angle = DirecetionEval(image)
    gatheredError += np.abs(angle - expectedAngle)
    print(f"Expected: {expectedAngle}, Gathered: {angle}")

averageError = gatheredError / len(programImages)
print(f"Average Error: {averageError}")