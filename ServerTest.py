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
    img = cv.imread("TrainingImages//perspectiveTest4.jpg")
    if img is None:
        return jsonify({"status": "error", "message": "Failed to read image"}), 400


    img = rectify(img)

    imgOrg = img.copy()

    # Pre-Processing
    figureImg = rectify(imgOrg)

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
    isUp, dotHeight, brickHeight, brickWidth = Matrix.find_up(blob, whole_blob)
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

    for y, row in enumerate(colorMatrix):
        new_row = []
        for x, col in enumerate(row):
            newBrick = brick.copy()
            newBrick["color"] = col

            midY = blob_brick_matrix[y][x].shape[0] // 2
            midX = blob_brick_matrix[y][x].shape[1] // 2
            if blob_brick_matrix[y][x][midY][midX] == 0:
                newBrick["isEmpty"] = True

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

    '''json_str = json.dumps(final_brick_matrix)
    with open("sample.json", "w") as f:
        f.write(json_str)'''

    return jsonify(final_brick_matrix)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000)
