import cv2 as cv
import numpy as np

# image = cv.imread('fishImages/fish1/fish1_1.jpg')
# image = cv.resize(image,(1008,756))
# image = image[270:500,220:800] #cropout of fish
# image = image[28:59,58:178] #cropout of block

def DominantColorsFun(img):
    # createBars funktion laver en bar med farverne og deres farvekoder. når scriptet er testet færdig skal den ændres til kun at returnere farvekode
    def createBars(color):
        red, green, blue = int(color[0]), int(color[1]), int(color[2])  # correct from bgr to rgb (except rn it's still bgr)
        return (red, green, blue)

    height, width, _ = np.shape(img)  # height and width required for calculating pixels
    crop = 0.4
    img = img[int(0+height*crop):int(height-height*crop), int(0+width*crop):int(width-width*crop)]
    height, width, _ = np.shape(img)
    data = np.reshape(img,(height * width, 3)) #get all pixel values into a list
    data = np.float32(data) #convert to float

    clusterAmount = 3 #amount of colors to find
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0) #maximum iterations and desired accuracy for criteria
    flags = cv.KMEANS_RANDOM_CENTERS #KMEANS_RANDOM/PP_CENTERS
    compactness, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, flags=flags)

    rgbValues = []

    for index, row in enumerate(centers):
        rgb = createBars(row)
        rgbValues.append(rgb)

    return(min(rgbValues)) #find darkest color. takes care of highlights

# #Quantize image
# centers = np.uint8(centers)
# imgQuantized = centers[labels.flatten()]
# imgQuantized = imgQuantized.reshape(img.shape)
#
# cv.imshow('Dominant colors', imgBar)
# cv.imshow('Image colors quantized', imgQuantized)
# cv.imshow('Image', image)
#
# cv.waitKey(0)