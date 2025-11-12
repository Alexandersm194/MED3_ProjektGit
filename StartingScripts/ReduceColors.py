import cv2 as cv
import numpy as np
from cv2 import waitKey

img = cv.imread('C:/Users/pauls/PycharmProjects/MED3_ProjektGit/fishImages/CoppedImages/image.png')
img = cv.resize(img,(1008,756))


z = img.reshape((-1,3))

# convert to np.float32
z = np.float32(z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv.kmeans(z,K,None,criteria,10,cv.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


cv.imshow('image', res2)
cv.waitKey()