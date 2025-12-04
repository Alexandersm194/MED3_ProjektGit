import cv2
import cv2 as cv
import numpy as np
import ModelDirection as MD
import Matrix
import Segmentation
import PreProcessing
import json
from DominantColors import DominantColorsFun
from ColorProcessor import visualizeMatrix, connectColors
from PointImgCrop import rectify
from BackgroundSubtraction import remove_background
from BrickClassifier import classify_brick_hist, classify_brick_size
from ThresholdTrainer import clusters_to_hist, train_color_histograms as trained_histograms
from BrickDetector import brick_detect

img = cv.imread("TestImages/Angle/0 degrees/AFig1.jpg")
#img = rectify(img)
imgOrg = img.copy()
#figureImg = imgOrg[]

yIn = img.shape[0] / 5.4
xIn = img.shape[1] / 8

# convert to integers for cropping
yIn = int(yIn)
xIn = int(xIn)

#figureImg = imgOrg[:yIn, :xIn]
figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]
cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.namedWindow("Org", cv.WINDOW_NORMAL)
cv.imshow("Image", img)
cv.imshow("Org", figureImg)
cv.waitKey(0)

#Background Removal
'''whole_blob = Segmentation.background_removal(img)[0]
blob = Segmentation.background_removal(figureImg)[0]'''

whole_blob = remove_background(img)
blob = remove_background(figureImg)
edge = MD.brickEdge(figureImg)[1]

cv.namedWindow("wholeBlob", cv.WINDOW_NORMAL)
cv.imshow("wholeBlob", whole_blob)
cv.waitKey(0)

#Direction
dominant_angle = MD.dominant_angle_from_lines(edge)

#Rotate
rotated = MD.rotateImage(blob, dominant_angle)
rotated_org = MD.rotateImage(figureImg, dominant_angle)

#FindBoundingBox
cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated)
cropped_org = rotated_org[y:y + h, x:x + w]

cv.imshow("Rotated", cropped_bin)
cv.waitKey(0)

#FindUp
isUp, dotHeight, brickHeight, brickWidth = Matrix.find_up(cropped_bin, whole_blob)
corrected_img_bin = cropped_bin
corrected_img = cropped_org
if isUp is False:
    corrected_img_bin = MD.rotateImage(cropped_bin, 180)
    corrected_img = MD.rotateImage(corrected_img, 180)

cv.imshow("corrected BINARY", corrected_img_bin)
cv.imshow("corrected", corrected_img)
cv.waitKey(0)
#BrickMatrix
brickWidth += int(brickHeight*0.05)
bricks = brick_detect(corrected_img, corrected_img_bin, brickWidth, brickHeight, dotHeight)

'''brickMat = []
for row in bricks:
    newRow = []
    for brick in row:
        if(brick is not None):
            newRow.append(1)
            cv.imshow("brick", brick)
            cv.waitKey(0)
        else:
            newRow.append(0)

    brickMat.append(newRow)'''

brickDic = {
    "size": 0,
    "color": "unknown"
}
tHist = trained_histograms()
finalBrickMat = []
for row in bricks:
    newRow = []
    for brick in row:
        if(brick is not None):
            newBrick = brickDic.copy()
            clusters = DominantColorsFun(brick)
            if not isinstance(clusters, list):
                clusters = [clusters]

            # Convert clusters to normalized HSV histogram
            hist = clusters_to_hist(clusters)

            # Classify brick based on trained histograms
            predicted_color = classify_brick_hist(hist, tHist)
            newBrick["color"] = predicted_color
            newBrick["size"] = classify_brick_size(brick)
            newRow.append(newBrick)
        else:
            newRow.append(None)

    finalBrickMat.append(newRow)
print(finalBrickMat)
cv.waitKey(0)
brick_matrix = Matrix.matrix_slice(corrected_img, brickHeight, brickWidth, dotHeight)
blob_brick_matrix = Matrix.matrix_slice(corrected_img_bin, brickHeight, brickWidth, dotHeight)



colorMatrix = []
tHist = trained_histograms()
for y, row in enumerate(brick_matrix):
    rows = []
    for x, col in enumerate(row):
        # Get all cluster colors
        clusters = DominantColorsFun(col)
        if not isinstance(clusters, list):
            clusters = [clusters]

        # Convert clusters to normalized HSV histogram
        hist = clusters_to_hist(clusters)

        # Classify brick based on trained histograms
        predicted_color = classify_brick_hist(hist, tHist)

        rows.append(predicted_color)
    colorMatrix.append(rows)

for y in range(len(colorMatrix)):
    print(colorMatrix[y])
cv2.waitKey(0)
colorMatrixImg, colorMatrix = connectColors(colorMatrix)

cv.imshow("colorMatrix", colorMatrixImg)
cv.waitKey(0)

# print(colorMatrix)
# print(len(brick_matrix))

#Feature Extraction And Classification
# for y, row in enumerate(brick_matrix):
#     for x, col in enumerate(row):
#         print(col.shape)
#         cv.imshow(f"{y},{x}", col)
#         cv.waitKey(0)
#         # cv.destroyAllWindows()

SomethingMatrix = []
# Brick
'''for y, row in enumerate(blob_brick_matrix):
    newRow = []
    for x, col in enumerate(row):
        midY = col.shape[0] // 2
        midX = col.shape[1] // 2
        if col[midY][midX] == 0:
            newRow.append("empty")
        else:
            newRow.append("Something")
    SomethingMatrix.append(newRow)

print(SomethingMatrix)'''


ColorMatrix = [
    ["empty", "empty" , "empty", "empty", "blue", "blue", "empty", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "blue", "blue", "blue", "blue", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "green", "green", "green", "green", "empty", "empty"],
    ["empty", "empty" , "green", "green", "green", "green", "blue", "blue", "empty", "empty"],
    ["empty", "empty" , "green", "empty", "green", "green", "empty", "blue", "empty", "empty"],
]


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


'''final_brick_matrix = []
for y, row in enumerate(ColorMatrix):
    final_brick_matrix.append([])

brick_final = {
    "length": 0,
    "color": "",
    "isEmpty": False
}

for y, row in enumerate(brick_matrix):
    curr_brick_color = row[0]["color"]
    curr_brick_isEmpty = row[0]["isEmpty"]
    curr_brick_length = 1

    for x in range(1, len(row)):
        if row[x]["isEmpty"]:
            new_brick = brick_final.copy()
            new_brick["length"] = 1
            new_brick["color"] = curr_brick_color
            new_brick["isEmpty"] = True

        else:
            if row[x]["color"] == curr_brick_color:
                curr_brick_length += 1
            else:
                new_brick = brick_final.copy()
                new_brick["length"] = curr_brick_length
                new_brick["color"] = curr_brick_color
                new_brick["isEmpty"] = brick_matrix[y][x]["isEmpty"]
                final_brick_matrix[y].append(new_brick)
                curr_brick_color = row[x]["color"]
                curr_brick_length = 1

    new_brick = brick_final.copy()
    new_brick["length"] = curr_brick_length
    new_brick["color"] = curr_brick_color
    new_brick["isEmpty"] = False
    final_brick_matrix[y].append(new_brick)



print(final_brick_matrix)'''

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


json_str = json.dumps(final_brick_matrix)
with open("sample.json", "w") as f:
    f.write(json_str)