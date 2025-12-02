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

def imageProcess(img):
    img = rectify(img)

    imgOrg = img.copy()

    # Pre-Processing
    # figureImg = rectify(imgOrg)
    yIn = img.shape[0] // 4
    xIn = img.shape[1] // 6
    figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]
    # Background Removal
    whole_blob = Segmentation.background_removal(img)[0]
    blob = Segmentation.background_removal(figureImg)[0]

    '''whole_blob = remove_background(img)
    blob = remove_background(figureImg)'''
    edge = MD.brickEdge(figureImg)[1]

    # Direction
    dominant_angle = MD.dominant_angle_from_lines(edge)

    # Rotate
    rotated = MD.rotateImage(blob, dominant_angle)
    rotated_org = MD.rotateImage(figureImg, dominant_angle)

    # FindBoundingBox
    cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated)
    cropped_org = rotated_org[y:y + h, x:x + w]

    # FindUp
    isUp, dotHeight, brickHeight, brickWidth = Matrix.find_up(cropped_bin, whole_blob)
    corrected_img_bin = cropped_bin
    corrected_img = cropped_org
    if isUp is False:
        corrected_img_bin = MD.rotateImage(cropped_bin, 180)
        corrected_img = MD.rotateImage(corrected_img, 180)

    # BrickMatrix
    brick_matrix = Matrix.matrix_slice(corrected_img, brickHeight, brickWidth, dotHeight)
    blob_brick_matrix = Matrix.matrix_slice(corrected_img_bin, brickHeight, brickWidth, dotHeight)

    colorMatrix = []
    for y, row in enumerate(brick_matrix):
        rows = []
        for x, col in enumerate(row):
            rows.append(DominantColorsFun(col))
        colorMatrix.append(rows)

    for y in range(len(colorMatrix)):
        print(colorMatrix[y])

    colorMatrixImg, colorMatrix = connectColors(colorMatrix)

    brick = {
        "color": None,
        "isEmpty": False,
    }

    brick_matrix = []

    patch_size = 10  # Size of the patch to check around the center (3x3)
    threshold = 0.5  # Fraction of pixels that must be black to consider it empty

    for y, row in enumerate(colorMatrix):
        new_row = []
        for x, col in enumerate(row):
            newBrick = brick.copy()
            newBrick["color"] = col

            blob = blob_brick_matrix[y][x]
            h, w = blob.shape

            # Define center patch coordinates
            midY, midX = h // 2, w // 2
            y1 = max(midY - patch_size // 2, 0)
            y2 = min(midY + patch_size // 2 + 1, h)
            x1 = max(midX - patch_size // 2, 0)
            x2 = min(midX + patch_size // 2 + 1, w)

            patch = blob[y1:y2, x1:x2]

            # Decide if empty based on patch
            if np.mean(patch == 0) >= threshold:
                newBrick["isEmpty"] = True
            else:
                newBrick["isEmpty"] = False

            new_row.append(newBrick)
        brick_matrix.append(new_row)

    brick_final = {
        "length": 0,
        "color": "",
        "isEmpty": False
    }

    final_brick_matrix = []

    for y, row in enumerate(brick_matrix):
        final_row = []
        curr_brick_color = None
        curr_brick_length = 0

        for brick in row:
            if brick["isEmpty"]:
                # Finish any current non-empty sequence
                if curr_brick_length > 0:
                    new_brick = brick_final.copy()
                    new_brick["length"] = curr_brick_length
                    new_brick["color"] = curr_brick_color
                    new_brick["isEmpty"] = False
                    final_row.append(new_brick)
                    curr_brick_color = None
                    curr_brick_length = 0

                # Add empty brick as its own entry
                new_brick = brick_final.copy()
                new_brick["length"] = 1
                new_brick["color"] = brick["color"]
                new_brick["isEmpty"] = True
                final_row.append(new_brick)

            else:
                # Non-empty brick
                if brick["color"] == curr_brick_color:
                    curr_brick_length += 1
                else:
                    # Finish previous sequence
                    if curr_brick_length > 0:
                        new_brick = brick_final.copy()
                        new_brick["length"] = curr_brick_length
                        new_brick["color"] = curr_brick_color
                        new_brick["isEmpty"] = False
                        final_row.append(new_brick)

                    # Start new sequence
                    curr_brick_color = brick["color"]
                    curr_brick_length = 1

        # Finish any remaining sequence at end of row
        if curr_brick_length > 0:
            new_brick = brick_final.copy()
            new_brick["length"] = curr_brick_length
            new_brick["color"] = curr_brick_color
            new_brick["isEmpty"] = False
            final_row.append(new_brick)

        final_brick_matrix.append(final_row)

    print(final_brick_matrix)

imageDir = "TestImages//Angle//45 degrees"
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
        cv2.imwrite(os.path.join(imageDir, testImgFile[i]), rect)

    if success:
        SuccesfullRect += 1
    else:
        FailRect += 1
    cv2.namedWindow("Rect", cv2.WINDOW_NORMAL)
    cv2.imshow("Rect", rect)
    cv2.waitKey(0)

print(f"Succesfull: {SuccesfullRect}, Fail: {FailRect}")
