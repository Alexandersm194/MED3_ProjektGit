import cv2 as cv
import numpy as np

chessIMG = cv.imread('TrainingImages/chessboard.jpg')
chessIMG = cv.resize(chessIMG,(640,480))

#select coordinates
tl = (270, 50)
bl = (80, 240)
tr = (555, 140)
br = (410, 380)

cv.circle(chessIMG, tl, 5, (0,0,255), 2)
cv.circle(chessIMG, bl, 5, (0,0,255), 2)
cv.circle(chessIMG, tr, 5, (0,0,255), 2)
cv.circle(chessIMG, br, 5, (0,0,255), 2)

#apply transformation
pointsOriginal = np.float32([tl, bl, tr, br])
pointsTransformed = np.float32([(0,0), (0,480), (640,0), (640,480)])

matrix = cv.getPerspectiveTransform(pointsOriginal, pointsTransformed)
chessTransformed = cv.warpPerspective(chessIMG, matrix, (640, 480))

cv.imshow('Original', chessIMG)
cv.imshow('Transformed', chessTransformed)
cv.waitKey(0)