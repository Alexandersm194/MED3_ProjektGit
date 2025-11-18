import cv2
import cv2 as cv
import numpy as np
import ModelDirection as MD
import Matrix
import Segmentation
import PreProcessing
import json
from DominantColors import DominantColorsFun
from ColorProcessor import visualizeMatrix

img = cv.imread("TrainingImages//test3.png")
imgOrg = img.copy()

#Pre-Processing


#Background Removal
blob = MD.blob(img)[0]
edge = MD.brickEdge(img)[1]

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", blob)
cv2.waitKey(0)

#Direction
dominant_angle = MD.dominant_angle_from_lines(edge)[1]

#Rotate
rotated = MD.rotateImage(blob, dominant_angle)
rotated_org = MD.rotateImage(imgOrg, dominant_angle)

#FindBoundingBox
cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated)
cropped_org = rotated_org[y:y + h, x:x + w]

cv2.imshow("Rotated", cropped_bin)
cv2.waitKey(0)

#FindUp
isUp, dotHight, brickHight, brickWidth = Matrix.find_up(blob, blob)
corrected_img_bin = cropped_bin
corrected_img = cropped_org
if isUp:
    corrected_img_bin = MD.rotateImage(cropped_bin, 180)
    corrected_img = MD.rotateImage(corrected_img, 180)

cv.imshow("corrected BINARY", corrected_img_bin)
cv.imshow("corrected", corrected_img)
cv.waitKey(0)

#BrickMatrix
brick_matrix = Matrix.matrix_slice(corrected_img, brickHeight, brickWidth, dotHeight)

colorMatrix = []
for y, row in enumerate(brick_matrix):
    rows = []
    for x, col in enumerate(row):
        rows.append(DominantColorsFun(col))
    colorMatrix.append(rows)

for y in range(len(colorMatrix)):
    print(colorMatrix[y])

visualizeMatrix(colorMatrix)

print(len(brick_matrix))

#Feature Extraction And Classification
'''for y, row in enumerate(brick_matrix):
    for x, col in enumerate(row):
        print(col.shape)
        cv.imshow(f"{y},{x}", col)
        cv.waitKey(0)
        cv.destroyAllWindows()'''

# Brick

ColorMatrix = [
    ["empty", "empty" , "empty", "empty", "blue", "blue", "empty", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "blue", "blue", "blue", "blue", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "green", "green", "green", "green", "empty", "empty"],
    ["empty", "empty" , "green", "green", "green", "green", "blue", "blue", "empty", "empty"],
    ["empty", "empty" , "green", "empty", "green", "green", "empty", "blue", "empty", "empty"],
]

final_brick_matrix = []
for y, row in enumerate(ColorMatrix):
    final_brick_matrix.append([])

brick = {
    "length": 0,
    "color": ""
}

for y, row in enumerate(ColorMatrix):
    curr_brick_color = row[0]
    curr_brick_length = 1

    for x in range(1, len(row)):
        if row[x] == curr_brick_color:
            curr_brick_length += 1
        else:
            new_brick = brick.copy()
            new_brick["length"] = curr_brick_length
            new_brick["color"] = curr_brick_color
            final_brick_matrix[y].append(new_brick)
            curr_brick_color = row[x]
            curr_brick_length = 1

    new_brick = brick.copy()
    new_brick["length"] = curr_brick_length
    new_brick["color"] = curr_brick_color
    final_brick_matrix[y].append(new_brick)

json_str = json.dumps(final_brick_matrix)
with open("sample.json", "w") as f:
    f.write(json_str)




