import cv2
import os
from flask import Flask, request, jsonify
from Main import LegoFigureProgram

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
    img = cv2.imread("TestImagesCropped//Lighting//Dark//JFig3.jpg")

    if img is None:
        return jsonify({"status": "error", "message": "Failed to read image"}), 400

    finalBrickMat = LegoFigureProgram(img)
    '''imgOrg = img.copy()
    # figureImg = imgOrg[]

    yIn = img.shape[0] / 5.4
    xIn = img.shape[1] / 8

    # convert to integers for cropping
    yIn = int(yIn)
    xIn = int(xIn)

    # figureImg = imgOrg[:yIn, :xIn]
    figureImg = imgOrg[yIn:img.shape[0] - yIn, xIn:img.shape[1] - xIn]
    # figureImg = rectify(img)[0]

    # Background Removal
    whole_blob = remove_background(img)
    blob = remove_background(figureImg)
    edge = MD.brickEdge(figureImg)[1]
    
    # Direction
    dominant_angle = MD.dominant_angle_from_lines(edge)

    # Rotate
    rotated_bin = MD.rotateImage(blob, dominant_angle)
    rotated_org = MD.rotateImage(figureImg, dominant_angle)

    # ----------------------------------------------------------
    # 2) Initial bounding box (required for find_up)
    # ----------------------------------------------------------
    cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
    cropped_org = rotated_org[y:y + h, x:x + w]

    # ----------------------------------------------------------
    # 3) Determine orientation from the first bounding box
    # ----------------------------------------------------------
    isUp, dotHeight, brickHeight, brickWidth, isOnSide = Matrix.find_up(cropped_bin, whole_blob)
    print("isOnSide:", isOnSide)

    # ----------------------------------------------------------
    # 4) If on side (90°) → rotate the FULL images and re-crop
    # ----------------------------------------------------------
    if isOnSide:
        # Rotate the full original-rotated images, NOT the cropped ones
        rotated_bin = MD.rotateImage(rotated_bin, 90)
        rotated_org = MD.rotateImage(rotated_org, 90)

        # New bounding box after 90° rotation
        cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
        cropped_org = rotated_org[y:y + h, x:x + w]

        # Orientation check again from fresh crop
        isUp = Matrix.find_up(cropped_bin, whole_blob)[0]

    # ----------------------------------------------------------
    # 5) If upside down (180°) → rotate FULL images again and re-crop
    # ----------------------------------------------------------
    if not isUp:
        rotated_bin = MD.rotateImage(rotated_bin, 180)
        rotated_org = MD.rotateImage(rotated_org, 180)

        # Fresh bounding box after 180° rotation
        cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated_bin)
        cropped_org = rotated_org[y:y + h, x:x + w]


    # ----------------------------------------------------------
    # 7) Brick detection
    # ----------------------------------------------------------
    brickWidth += int(brickHeight * 0.05)
    bricks = brick_detect(cropped_org, cropped_bin, brickWidth, brickHeight, dotHeight)

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
                if not isinstance(clusters, list):
                    clusters = [clusters]

                hist = clusters_to_hist(clusters)

                predicted_color = classify_brick_mahalanobis(hist, tHist)
                newBrick["color"] = predicted_color
                newBrick["size"] = classify_brick_size(brick, brickHeight, brickWidth)
                newRow.append(newBrick)
            else:
                newRow.append(None)

        finalBrickMat.append(newRow)'''
    return jsonify(finalBrickMat)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000)
