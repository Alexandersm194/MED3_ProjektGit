import cv2 as cv

'''fra her: https://www.youtube.com/watch?v=ezeZoUqne54'''

img_path = cv.imread('C:/Users/magnu/PycharmProjects/MED3_ProjektGit/fishImages/fish1/fish1_9.jpg')

img = cv.resize(img_path, None, fx=0.25, fy=0.25)

x, y, w, h = cv.selectROI('Select ROI', img, showCrosshair=True, fromCenter=False)

print(f"{x = }, {y = }, {w = }, {h = }")

cropped_img = img[y:y+h, x:x+w]

cv.imshow('Cropped Image', cropped_img)
cv.waitKey(0)

cv.destroyAllWindows()

# cv.imshow('img', img)
# cv.waitKey(0)