import cv2 as cv
import numpy as np
import ModelDirection as MD
import Matrix
import Segmentation

img = cv.imread("C://Users//Alexa//Documents//GitHub//MED3_ProjektGit//TrainingImages//Fisk.jpg")
imgOrg = img.copy()

#Pre-Processing


#Background Removal
blob = MD.blob(img)[0]
edge = MD.brickEdge(img)[1]

#Direction
dominant_angle = MD.dominant_angle_from_lines(edge)[1]

#Rotate
rotated = MD.rotateImage(blob, dominant_angle)
rotated_org = MD.rotateImage(imgOrg, dominant_angle)

#FindBoundingBox
cropped_bin, x, y, w, h = Segmentation.find_bounding_box(rotated)
cropped_org = rotated_org[y:y + h, x:x + w]

#FindUp
isUp, dotHight, brickHight, brickWidth = Matrix.find_up(cropped_bin)
corrected_img_bin = cropped_bin
corrected_img = cropped_org
if not isUp:
    corrected_img_bin = MD.rotateImage(cropped_bin, 180)
    corrected_img = MD.rotateImage(corrected_img, 180)

cv.imshow("corrected BINARY", corrected_img_bin)
cv.imshow("corrected", corrected_img)
cv.waitKey(0)

#BrickMatrix
brick_matrix = Matrix.matrix_slice(corrected_img, brickHight, brickWidth, dotHight)

#Feature Extraction And Classification
for y, row in enumerate(brick_matrix):
    for x, col in enumerate(row):
        print(col.shape)
        cv.imshow(f"{y},{x}", col)
        cv.waitKey(0)
        cv.destroyAllWindows()




