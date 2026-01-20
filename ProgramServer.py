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
    img = cv2.imread("TestImagesV2Cropped//Optimal//AFig3.jpg")

    if img is None:
        return jsonify({"status": "error", "message": "Failed to read image"}), 400

    finalBrickMat = LegoFigureProgram(img)

    return jsonify(finalBrickMat)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000)
