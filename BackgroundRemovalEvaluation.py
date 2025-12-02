import cv2
import Segmentation

programImg = cv2.imread("debug_edged.jpg")
groundTruthImg = cv2.imread("groundTruth.jpg")

programImg = Segmentation.background_removal(programImg)

groundTruthImg = cv2.threshold(groundTruthImg, 10, 255, cv2.THRESH_BINARY)[1]

intersection = 0
falseNegative = 0
for y, row in enumerate(groundTruthImg):
    for x, pixel in enumerate(row):
        if groundTruthImg[y][x] == programImg[y][x] and programImg[y][x] != 0:
            intersection += 1
        elif programImg[y][x] != programImg[y][x]:
            falseNegative += 1

