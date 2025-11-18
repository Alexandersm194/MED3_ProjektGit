import cv2 as cv
import numpy as np

'''cv crop'''
img_path = cv.imread('C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish2/fish2_2.jpg')

img = cv.resize(img_path, None, fx=0.25, fy=0.25)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (9, 9), 0)
_, thresh = cv.threshold(blur, 160, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = np.ones((11, 11), np.uint64)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
edge = cv.Canny(opening, 99, 100)

cnts = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

c = max(cnts, key=cv.contourArea)

x, y, w, h = cv.boundingRect(c)

crop = img[y:y+h, x:x+w].copy()

cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

corners = cv.goodFeaturesToTrack(gray, 25, 0.1, 50)
corners = np.int64(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(crop, (x, y), 3, 255, -1)

cv.imshow('img', img)
cv.imshow('crop', crop)
cv.waitKey(0)