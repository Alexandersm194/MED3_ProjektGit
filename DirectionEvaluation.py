import cv2 as cv
import numpy as np
import ModelDirection as MD
import os
from PointImgCrop import rectify


def DirecetionEval(img):
    yIn = int(img.shape[0] / 5.4)
    xIn = int(img.shape[1] / 8)
    figureImg = img[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]

    edge = MD.brickEdge(figureImg)[1]

    # detector returns a normal → convert to orientation
    dominant_angle = MD.dominant_angle_from_lines(edge)

    return dominant_angle

def norm180(a):
    return a % 180

def orientation_diff(a, b):
    d = abs(norm180(a) - norm180(b))
    return min(d, 180 - d)

def angle_error(det, exp):
    return orientation_diff(det, exp)

programDir = "TestImagesCropped//Lighting//HardLighting"
expectedAngle = 0
programImages = []
figureNames = []

if os.path.isdir(programDir):
    for file in os.listdir(programDir):
        full_path = os.path.join(programDir, file)
        img = cv.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            programImages.append(img)
            figureNames.append(file)

gatheredError = 0

for i, image in enumerate(programImages):
    det = DirecetionEval(image)

    error = angle_error(det, expectedAngle)
    if det > 90:
        error = 90 - error
    gatheredError += error

    print(f"{error:.2f}")

averageError = gatheredError / len(programImages)
print(f"Average Error: {averageError}")