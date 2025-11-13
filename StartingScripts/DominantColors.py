import cv2 as cv
import numpy as np

#createBars funktion laver en bar med farverne og deres farvekoder. når scriptet er testet færdig skal den ændres til kun at returnere farvekode
def createBars(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar [:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0]) #correct from bgr to rgb
    return bar, (red, green, blue)

img = cv.imread('C:/Users/pauls/PycharmProjects/MED3_ProjektGit/fishImages/fish1/fish1_1.jpg')
img = cv.resize(img,(1008,756))
img = img[270:500,220:800] #cropout of fish
img = img[28:59,58:178] #cropout of block
height, width, _ = np.shape(img) #height and width required for calculating pixels

data = np.reshape(img,(height * width, 3)) #get all pixel values into a list
data = np.float32(data) #convert to float

clusterAmount = 3 #amount of colors to find
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0) #maximum iterations and desired accuracy for criteria
flags = cv.KMEANS_RANDOM_CENTERS #KMEANS_RANDOM/PP_CENTERS
compactness, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, flags=flags)

#til createBars function
font = cv.FONT_HERSHEY_SIMPLEX
bars = []
rgbValues = []

for index, row in enumerate(centers):
    bar, rgb = createBars(180, 180, row)
    bars.append(bar)
    rgbValues.append(rgb)

imgBar = np.hstack(bars)

for index, row in enumerate(rgbValues):
    image = cv.putText(imgBar, f'RGB: {row}', (5 + 180 * index, 180 - 10), font, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    # print(f'{index + 1}. RGB: {row}')

print(min(rgbValues)) #find darkest color. takes care of highlights

#Quantize image
centers = np.uint8(centers)
imgQuantized = centers[labels.flatten()]
imgQuantized = imgQuantized.reshape(img.shape)

cv.imshow('Dominant colors', imgBar)
cv.imshow('Image colors quantized', imgQuantized)
cv.imshow('Image', img)

cv.waitKey(0)