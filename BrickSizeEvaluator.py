import Main
from BrickSizeGroundTruth import getGroundTruth
import os
import cv2
import numpy as np

GroundTruths = getGroundTruth()


imageDir = "TestImagesV1//Lighting//Optimal"
images = []
figureNames = []


if os.path.isdir(imageDir):
    for file in os.listdir(imageDir):
        full_path = os.path.join(imageDir, file)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Could not load image: {full_path}")
        else:
            print(f"Image loaded successfully: {full_path}")
            images.append(img)
            figureNames.append(file)


TP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
FP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
FN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

VerticalError = []
HorizontalError = []

for i, image in enumerate(images):
    ProgramResult = Main.LegoFigureProgram(image)
    GroundTruth = GroundTruths[i]

    # Store size mismatch errors
    VerticalError.append(len(GroundTruth) - len(ProgramResult))
    HorizontalError.append(len(GroundTruth[0]) - len(ProgramResult[0]))

    for y, row in enumerate(GroundTruth):
        for x, col in enumerate(row):

            # ---- 1. Bounds check ----
            if y >= len(ProgramResult) or x >= len(ProgramResult[y]):
                # ProgramResult too small → algorithm missed a brick
                if col is None:
                    TP[0] += 1   # both "nothing"? ground truth = None and predicted out of bounds = None
                else:
                    FN[col] += 1
                continue

            # Safe to access now
            brick = ProgramResult[y][x]
            matrix[brick][col] += 1
            # ---- 2. Normal comparison logic ----
            if col is None and brick is None:
                TP[0] += 1
                continue

            if col is None and brick is not None:
                predicted_size = brick["size"]
                FP[predicted_size] += 1
                continue

            if col is not None and brick is None:
                FN[col] += 1
                continue

            predicted_size = brick["size"]

            if col == predicted_size:
                TP[col] += 1
            else:
                FN[col] += 1
                FP[predicted_size] += 1

print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")

precisions = []
recalls = []
f1Scores = []

for i in range(len(TP)):
    if (TP[i] + FN[i]) == 0 or (TP[i] + FP[i]) == 0:
        continue
    precision = TP[i] / (TP[i] + FP[i])
    precisions.append(precision)
    recall = TP[i] / (TP[i] + FN[i])
    recalls.append(recall)
    if precision + recall == 0:
        f1Score = 0
    else:
        f1Score = 2 * (precision * recall) / (precision + recall)
    f1Scores.append(f1Score)

    print(f"Length {i}: Precision: {precision}, Recall: {recall}, F1Score: {f1Score}")

precisions.sort()
recalls.sort()
f1Scores.sort()

print(f"Averages | Precision: {np.mean(precisions)}, Recall: {np.mean(recalls)}, F1Score: {np.mean(f1Scores)}")

print(VerticalError)
print(HorizontalError)

for row in matrix:
    print(row)
