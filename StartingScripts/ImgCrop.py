import cv2 as cv

img = cv.imread('C:/Users/magnu/PycharmProjects/MED3_ProjektGit/TrainingImages/backgroundtestCropped.jpg')


cv.imshow("crop", crop_img)
cv.waitKey(0)