import cv2
import Segmentation
import os

programDir = "TestImages//Direction//45 degrees"
groundDir = "TestImages//Direction//45 degrees GT"
programImages = []
groundImages = []
figureNames = []


if os.path.isdir(programDir):
    for file in os.listdir(programDir):
        full_path = os.path.join(programDir, file)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            programImages.append(img)
            figureNames.append(file)

if os.path.isdir(groundDir):
    for file in os.listdir(groundDir):
        full_path = os.path.join(groundDir, file)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            groundImages.append(img)
else:
    print("This is not a functional path!")

def IoU(programImg, groundTruthImg):
    h, w = programImg.shape[:2]
    groundTruthImg = cv2.resize(groundTruthImg, (w, h), interpolation=cv2.INTER_NEAREST)

    yIn = programImg.shape[0] // 4
    xIn = programImg.shape[1] // 6
    programImg = programImg[yIn:programImg.shape[0] - yIn, xIn:programImg.shape[1] - xIn]
    groundTruthImg = groundTruthImg[yIn:groundTruthImg.shape[0] - yIn, xIn:groundTruthImg.shape[1] - xIn]
        # Background Removal
    programImg = Segmentation.background_removal(programImg)[0]

    groundTruthImg = cv2.threshold(groundTruthImg, 1, 255, cv2.THRESH_BINARY)[1]

    intersection = 0.000
    falseNegative = 0.000
    for y, row in enumerate(groundTruthImg):
        for x, pixel in enumerate(row):
            if groundTruthImg[y][x] == programImg[y][x] and programImg[y][x] != 0:
                intersection += 1
            elif programImg[y][x] != groundTruthImg[y][x]:
                falseNegative += 1
    iou = intersection/(intersection + falseNegative)
    if iou < 0.6:
        cv2.namedWindow("Program", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)
        cv2.imshow("Program", programImg)
        cv2.imshow("Ground Truth", groundTruthImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return iou

gatheredIoU = 0
for i, img in enumerate(programImages):
    iou = IoU(programImages[i], groundImages[i])
    print(f"Figure {figureNames[i]}: {iou}")
    gatheredIoU += iou

averageIoU = gatheredIoU / len(programImages)
print(f"Average: {averageIoU}")
