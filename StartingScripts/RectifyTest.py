import cv2 as cv
import numpy as np

chessIMG = cv.imread('TrainingImages/chessboard.jpg')
chessIMG = cv.resize(chessIMG,(0,0), fx=0.5, fy=0.5)
chessIMGGray = cv.cvtColor(chessIMG, cv.COLOR_BGR2GRAY)

finalWidth = 600
finalHeight = 600

#initialize lowest x and y value for corners
# xMin, xMax, yMin, yMax = 1000, 0, 1000, 0

#draw cicles at coordinates
# cv.circle(chessIMG, tl, 5, (0,0,255), 2)
# cv.circle(chessIMG, bl, 5, (0,0,255), 2)
# cv.circle(chessIMG, tr, 5, (0,0,255), 2)
# cv.circle(chessIMG, br, 5, (0,0,255), 2)


#goodFeaturesToTrack detects corners
corners = cv.goodFeaturesToTrack(chessIMGGray, 100, 0.12, 10)
corners = np.intp(corners)

#select coordinates
pointX, pointY = 500, 500
tl = (pointX, pointY)
bl = (pointX, pointY)
tr = (pointX, pointY)
br = (pointX, pointY)


for corner in corners:
    x, y = corner.ravel() #ravel flattens the array
    if x < pointX:
        pointX = x
        bl = (x, y)
    # if x > xMax:
    #     xMax = x
    # if y < yMin:
    #     yMin = y
    # if y > yMax:
    #     yMax = y
    cv.circle(chessIMG, (x, y), 5, (0,0,255), 0)
    # print(x, y)

# print(xMin, xMax, yMin, yMax)
print(bl)


#apply transformation
pointsOriginal = np.float32([tl, bl, tr, br])
pointsTransformed = np.float32([(0,0), (0,finalHeight), (finalWidth,0), (finalWidth,finalHeight)])

matrix = cv.getPerspectiveTransform(pointsOriginal, pointsTransformed)
chessTransformed = cv.warpPerspective(chessIMG, matrix, (finalWidth, finalHeight))

cv.imshow('Original', chessIMG)
cv.imshow('Transformed', chessTransformed)
cv.waitKey(0)