# server_cv.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
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
from BrickDetector import brick_detect
from BrickClassifier import classify_brick_hist, classify_brick_size
from ThresholdTrainer import clusters_to_hist, train_color_histograms as trained_histograms

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/process_image", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    #img = cv2.imread(file_path)
    img = cv.imread("TestImages/Angle/0 degrees/MFig2.jpg")
    #img = cv.imread("TrainingImages//test.jpg")
    #img = cv.imread("uploads//photo.jpg")
    if img is None:
        return jsonify({"status": "error", "message": "Failed to read image"}), 400


    #img = rectify(img)

    imgOrg = img.copy()

    # Pre-Processing
    #figureImg = rectify(imgOrg)

    yIn = img.shape[0] / 5.4
    xIn = img.shape[1] / 8

    # convert to integers for cropping
    yIn = int(yIn)
    xIn = int(xIn)

    # figureImg = imgOrg[:yIn, :xIn]
    figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]


    # Background Removal
    '''whole_blob = Segmentation.background_removal(img)[0]
    blob = Segmentation.background_removal(figureImg)[0]'''

    whole_blob = remove_background(img)
    blob = remove_background(figureImg)
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
    brickWidth += int(brickHeight * 0.05)
    bricks = brick_detect(corrected_img, corrected_img_bin, brickWidth, brickHeight, dotHeight)

    brickDic = {
        "size": 0,
        "color": "unknown"
    }
    tHist = trained_histograms()
    finalBrickMat = []
    for row in bricks:
        newRow = []
        for brick in row:
            if (brick is not None):
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
    return jsonify(finalBrickMat)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000)
