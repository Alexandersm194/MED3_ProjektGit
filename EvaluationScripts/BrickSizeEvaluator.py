import Main
from BrickSizeGroundTruth import getGroundTruth
import os
import cv2
import numpy as np

GroundTruths = getGroundTruth()
imageDir = "TestImagesV2Cropped//Optimal"
images, figureNames = [], []

if os.path.isdir(imageDir):
    for file in os.listdir(imageDir):
        full_path = os.path.join(imageDir, file)
        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not load image: {full_path}")
        else:
            images.append(img)
            figureNames.append(file)

# ----------------------------
# PARAMETERS
# ----------------------------
NUM_CLASSES = 13  # 0 = empty, 1..12 = brick sizes
TP = [0] * NUM_CLASSES
FP = [0] * NUM_CLASSES
FN = [0] * NUM_CLASSES
matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
VerticalError, HorizontalError = [], []
skipped_images = 0

# ---- Safe increment helpers ----
def safe_inc(arr, idx):
    if isinstance(idx, int) and 0 <= idx < len(arr):
        arr[idx] += 1
    else:
        print(f"[WARN] Index out of range ({idx})")

def safe_mat_inc(gt, pr):
    if isinstance(gt, int) and 0 <= gt < NUM_CLASSES and \
       isinstance(pr, int) and 0 <= pr < NUM_CLASSES:
        matrix[gt][pr] += 1

for i, image in enumerate(images):
    ProgramResult = Main.LegoFigureProgram(image)
    GroundTruth = GroundTruths[i]

    # Check if detection failed
    if not ProgramResult or len(ProgramResult) == 0:
        print(f"[SKIP] Detection failed for {figureNames[i]}")
        skipped_images += 1

        for gt_row in GroundTruth:
            for gt_val in gt_row:
                if gt_val not in (None, 0):
                    safe_inc(FN, gt_val)
                    safe_mat_inc(gt_val, 0)

        VerticalError.append(len(GroundTruth))
        HorizontalError.append(len(GroundTruth[0]) if len(GroundTruth) > 0 else 0)
        continue

    gt_h = len(GroundTruth)
    pr_h = len(ProgramResult)
    gt_w = len(GroundTruth[0]) if gt_h > 0 else 0
    pr_w = len(ProgramResult[0]) if pr_h > 0 else 0
    VerticalError.append(gt_h - pr_h)
    HorizontalError.append(gt_w - pr_w)


    for y, gt_row in enumerate(GroundTruth):
        for x, gt_val in enumerate(gt_row):
            pr_val = None
            if y < pr_h and x < len(ProgramResult[y]):
                pr_val = ProgramResult[y][x]

            gt_brick, pr_brick = gt_val, pr_val

            if gt_brick is None and pr_brick is None:
                safe_inc(TP, 0)
                safe_mat_inc(0, 0)
                continue

            if gt_brick is None and pr_brick is not None:
                predicted = pr_brick["size"]
                safe_inc(FP, predicted)
                safe_mat_inc(0, predicted)
                continue

            if gt_brick not in (None, 0) and pr_brick is None:
                safe_inc(FN, gt_brick)
                safe_mat_inc(gt_brick, 0)
                continue

            if gt_brick not in (None, 0) and pr_brick is not None:
                predicted = pr_brick["size"]
                if predicted == gt_brick:
                    safe_inc(TP, gt_brick)
                else:
                    safe_inc(FN, gt_brick)
                    safe_inc(FP, predicted)
                safe_mat_inc(gt_brick, predicted)

print(f"\nSkipped images: {skipped_images} / {len(images)}")
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")

precisions, recalls, f1Scores = [], [], []

for size in range(1, NUM_CLASSES):
    if TP[size] + FN[size] == 0:
        continue

    tp, fp, fn = TP[size], FP[size], FN[size]

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1 = 0 if (precision + recall == 0) else 2 * (precision * recall) / (precision + recall)

    precisions.append(precision)
    recalls.append(recall)
    f1Scores.append(f1)

    print(f"Class {size}: Precision={precision:.3f}, Recall={recall:.3f}, F1Score={f1:.3f}")

avg_precision = np.mean(precisions) if precisions else 0
avg_recall    = np.mean(recalls) if recalls else 0
avg_f1        = np.mean(f1Scores) if f1Scores else 0

print(f"Averages | Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1Score: {avg_f1:.3f}")


print("\nVertical Error:", VerticalError)
print("Horizontal Error:", HorizontalError)

print("\nConfusion Matrix:")
for row in matrix:
    print(row)
