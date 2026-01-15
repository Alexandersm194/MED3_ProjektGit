import cv2 as cv
import ModelDirection as MD
import Matrix
import Segmentation
import json
from DominantColors import DominantColorsFun
from BackgroundSubtraction import remove_background
from BrickClassifier import classify_brick_size, classify_brick_mahalanobis
from ColorModelTrainer import cluster_to_feature, train_color_mahalanobis
from BrickDetector import brick_detect
from RectifyScript import rectify
from BrickTemplate import findReferenceBrick


def LegoFigureProgram(img):
    img = rectify(img) #RectifyScript.py
    imgOrg = img.copy()

    yIn = img.shape[0] / 5.4
    xIn = img.shape[1] / 8  #cropping the rectified img to not get the borders (instead of rectifying twice)

    yIn = int(yIn)
    xIn = int(xIn)

    figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn] #saving the cropped rectified img

    #BackgroundSubtraction.py
    whole_blob = remove_background(img)     #blob detection - all the blobs (inc. reference bricks) (after initial rectify)
    blob = remove_background(figureImg)     #blob detection - for just the figure
    edge = MD.brickEdge(figureImg)[1]       #ModelDirection.py -


    dominant_angle = MD.dominant_angle_from_lines(edge)

    rotated_bin = MD.rotateImage(blob, dominant_angle)
    rotated_org = MD.rotateImage(figureImg, dominant_angle)

    cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
    cropped_org = rotated_org[y:y + h, x:x + w]

    brickTemp, dotHeight, brickHeight, brickLength = findReferenceBrick(whole_blob)
    isUp, isOnSide = Matrix.find_up(cropped_bin, brickTemp)
    print("isOnSide:", isOnSide)

    if isOnSide:
        rotated_bin = MD.rotateImage(rotated_bin, 90)
        rotated_org = MD.rotateImage(rotated_org, 90)

        cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
        cropped_org = rotated_org[y:y + h, x:x + w]

        isUp = Matrix.find_up(cropped_bin, whole_blob)[0]

    if not isUp:
        rotated_bin = MD.rotateImage(rotated_bin, 180)
        rotated_org = MD.rotateImage(rotated_org, 180)

        cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
        cropped_org = rotated_org[y:y + h, x:x + w]

    brickLength += int(brickHeight * 0.05)
    bricks = brick_detect(cropped_org, cropped_bin, brickLength, brickHeight, dotHeight)

    brickDic = {
        "size": 0,
        "color": "unknown"
    }
    # tHist = trained_histograms()
    tHist = train_color_mahalanobis()
    finalBrickMat = []
    for row in bricks:
        newRow = []
        for brick in row:
            if (brick is not None):
                newBrick = brickDic.copy()

                clusters = DominantColorsFun(brick)

                hist = cluster_to_feature(clusters)

                predicted_color = classify_brick_mahalanobis(hist, tHist)
                newBrick["color"] = predicted_color
                newBrick["size"] = classify_brick_size(brick, brickHeight, brickLength)
                newRow.append(newBrick)
            else:
                newRow.append(None)

        finalBrickMat.append(newRow)

    json_str = json.dumps(finalBrickMat)
    with open("sample.json", "w") as f:
            f.write(json_str)

    return finalBrickMat

figure = LegoFigureProgram(cv.imread("TestImagesV2Cropped//Optimal//AFig3.jpg"))
for row in figure:
    print(row)
cv.waitKey(0)


