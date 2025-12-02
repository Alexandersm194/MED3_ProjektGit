import cv2
import Segmentation

programImg = cv2.imread("debug_edged.jpg")
groundTruthImg = cv2.imread("groundTruth.jpg")

programImg = Segmentation.background_removal(programImg)

