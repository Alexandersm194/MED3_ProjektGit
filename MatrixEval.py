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

#ground truths
AFig1 = [
    ["empty", "empty" , "empty", "yellow", "yellow", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"],
    ["empty", "yellow" , "yellow", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["red", "red" , "red", "red", "red", "red", "red", "red"],
    ["yellow", "yellow" , "yellow", "yellow", "blue", "blue", "blue", "blue"],
    ["yellow", "empty" , "empty", "empty", "empty", "empty", "empty", "lime"] ]

AFig2 = [
    ["empty", "empty", "blue", "blue", "empty", "empty"],
    ["empty", "empty", "orange", "orange", "empty", "empty"],
    ["empty", "red", "red", "red", "red", "empty"],
    ["yellow", "yellow", "yellow", "yellow", "yellow", "yellow"],
    ["blue", "blue", "blue", "blue", "blue", "blue"],
    ["empty", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["empty", "empty", "red", "red", "empty", "empty"],
    ["empty", "green", "green", "green", "green", "empty"]]

AFig3 = [
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "yellow", "yellow", "yellow", "yellow", "empty", "empty"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["blue", "blue" , "blue", "blue", "blue", "blue", "blue", "blue"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"]]

AFig4 = [
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "yellow", "yellow", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"],
    ["empty", "blue" , "blue", "blue", "blue", "blue", "blue", "empty"],
    ["red", "red" , "red", "red", "blue", "blue", "blue", "blue"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["empty", "blue" , "blue", "blue", "blue", "blue", "blue", "empty"],
    ["empty", "red" , "red", "empty", "empty", "yellow", "yellow", "empty"],
    ["empty", "red" , "red", "empty", "empty", "red", "red", "empty"]]

PFig4 = [
    ["empty", "empty", "empty", "empty", "yellow", "yellow", "empty", "empty", "yellow", "yellow", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "empty", "empty", "empty"],
    ["yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow"],
    ["empty", "red", "red", "red", "red", "orange", "orange", "orange", "orange", "red", "red", "red", "red", "empty"],
    ["empty", "empty", "empty", "black", "black", "black", "black", "black", "black", "black", "black", "empty", "empty", "empty"] ]



truthFig = AFig1
img = cv.imread("TestImages/Angle/0 degrees/AFig1.jpg")
imgOrg = img.copy()
#figureImg = imgOrg[]

yIn = img.shape[0] / 5.4
xIn = img.shape[1] / 8

# convert to integers for cropping
yIn = int(yIn)
xIn = int(xIn)

#figureImg = imgOrg[:yIn, :xIn]
figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]
# cv.namedWindow("Image", cv.WINDOW_NORMAL)
# cv.namedWindow("Org", cv.WINDOW_NORMAL)
# cv.imshow("Image", img)
# cv.imshow("Org", figureImg)
# cv.waitKey(0)

#Background Removal
'''whole_blob = Segmentation.background_removal(img)[0]
blob = Segmentation.background_removal(figureImg)[0]'''

whole_blob = remove_background(img)
blob = remove_background(figureImg)
edge = MD.brickEdge(figureImg)[1]

# cv.namedWindow("wholeBlob", cv.WINDOW_NORMAL)
# cv.imshow("wholeBlob", whole_blob)
# cv.waitKey(0)

#Direction
dominant_angle = MD.dominant_angle_from_lines(edge)

#Rotate
rotated = MD.rotateImage(blob, dominant_angle)
rotated_org = MD.rotateImage(figureImg, dominant_angle)

#FindBoundingBox
cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated)
cropped_org = rotated_org[y:y + h, x:x + w]

# cv.imshow("Rotated", cropped_bin)
# cv.waitKey(0)

#FindUp
isUp, dotHeight, brickHeight, brickWidth = Matrix.find_up(cropped_bin, whole_blob)
corrected_img_bin = cropped_bin
corrected_img = cropped_org
if isUp is False:
    corrected_img_bin = MD.rotateImage(cropped_bin, 180)
    corrected_img = MD.rotateImage(corrected_img, 180)

# cv.imshow("corrected BINARY", corrected_img_bin)
# cv.imshow("corrected", corrected_img)
# cv.waitKey(0)

#BrickMatrix
brickWidth += int(brickHeight*0.05) #changing brickWidth fixes cropping on this model but will not work with other models
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


cv.imshow("corrected", corrected_img)
cv.waitKey(0)

a = True
while a:
    print("how much to rotate clockwise?")
    rotation = input("type 0, 90, 180, or 270")
    if rotation == "0":
        rotation = 0
        a = False
    if rotation == "90":
        rotation = 3
        a = False
    if rotation == "180":
        rotation = 2
        a = False
    if rotation == "270":
        rotation = 1
        a = False

if rotation == 3:
    print("rotating 90")
    np.rot90(colorMatrix, 3)
if rotation == 2:
    print("rotating 180")
    np.rot90(colorMatrix, 2)
if rotation == 1:
    print("rotating 270")
    np.rot90(colorMatrix, 1)

print(colorMatrix)
cv.imshow("color", colorMatrixImg)
def brickColor(color):
    brick = np.zeros((100, 75, 3), np.uint8)  # zeros creates an ndarray of zeroes. 3rd shape value is amount of numbers in tuple. uint8 goes from 0 to 255 and is often used for images
    brick[:] = color  # assigns color value to each element in ndarray
    return brick

correct = 0
incorrect = 0
if len(colorMatrix) <= len(truthFig) and len(colorMatrix[0]) <= len(truthFig[0]):
    for y in range(len(colorMatrix)):
        for x in range(len(colorMatrix[y])):
            brick = brickColor(colorMatrix[y][x])
            cv.imshow(str(x), brick)
            cv.waitKey(50)
            cv.destroyWindow(str(x - 1))
            # print("is this color ", truthFig[y][x], "?")
            # answer = input("y/n")

            a = True
            b = ""
            while a:
                print("is this color ", truthFig[y][x], "?")
                b = input("y/n")

                if b == "y" or b == "n":
                    print(b)
                    a= False
                else:
                    print("Invalid input, please try again.")

            if b == "y":
                correct += 1
            if b == "n":
                incorrect += 1
print("correct:", correct)
print("incorrect:", incorrect)

if len(colorMatrix) == len(truthFig):
    print("same height")
else:
    print("height is ", len(colorMatrix) - len(truthFig), " bricks off")

if len(colorMatrix[0]) == len(truthFig[0]):
    print("same width")
else:
    print("width is ", len(colorMatrix[0]) - len(truthFig[0]), " bricks off")