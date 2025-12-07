import cv2
import numpy as np
import ModelDirection as MD
import Matrix
import Segmentation
import PreProcessing
import json
from DominantColors import DominantColorsFun
from ColorProcessor import visualizeMatrix, connectColors
from PointImgCrop import rectify
from AlexTestMatrices import brickColorMatrices, brickSizeMatrices
import os
import Main


imageDir = "TestImagesV1//Lighting//Dark"
newImageDir = "TestImagesCropped//Lighting//Dark"
testImages = []
testImgFile = []

if os.path.isdir(imageDir):
    for file in os.listdir(imageDir):
        full_path = os.path.join(imageDir, file)
        img = cv2.imread(full_path)
        testImgFile.append(file)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            testImages.append(img)
else:
    print("This is not a functional path!")

SuccesfullRect = 0
FailRect = 0
for i, img in enumerate(testImages):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    rect, success = rectify(img)

    if success:
        cv2.imwrite(os.path.join(newImageDir, testImgFile[i]), rect)

    if success:
        SuccesfullRect += 1
    else:
        FailRect += 1
    cv2.namedWindow("Rect", cv2.WINDOW_NORMAL)
    cv2.imshow("Rect", rect)
    cv2.waitKey(0)

print(f"Succesfull: {SuccesfullRect}, Fail: {FailRect}")
