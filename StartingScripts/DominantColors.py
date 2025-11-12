import cv2 as cv
import numpy as np

def createBars(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar [:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0]) #correct from bgr to rgb
    return bar, (red, green, blue)

img = cv.imread('C:/Users/pauls/PycharmProjects/MED3_ProjektGit/fishImages/CoppedImages/image.png')
# img = cv.resize(img,(1008,756))
# img = img[270:500,220:800]
height, width, _ = np.shape(img)

data = np.reshape(img,(height * width, 3)) #get all pixel values into a list
data = np.float32(data) #convert to float

clusterAmount = 4
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0) #maximum iterations and desired accuracy for criteria
flags = cv.KMEANS_PP_CENTERS
compactness, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, flags=flags)
print(centers)

font = cv.FONT_HERSHEY_SIMPLEX
bars = []
rgbValues = []

for index, row in enumerate(centers):
    bar, rgb = createBars(200, 200, row)
    bars.append(bar)
    rgbValues.append(rgb)

#could try to merge colors with similar rgb values
#if rgbValues[i] > rgbValues[0]+30 or rgbValues[i] < rgbValues[0]-30
imgBar = np.hstack(bars)

for index, row in enumerate(rgbValues):
    image = cv.putText(imgBar, f'RGB: {row}', (5 + 200 * index, 200 - 10), font, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    # print(f'{index + 1}. RGB: {row}')

cv.imshow('Dominant colors', imgBar)
cv.imshow('Image', img)

cv.waitKey(0)