import cv2 as cv
import numpy as np

# image = cv.imread('fishImages/fish1/fish1_1.jpg')
# image = cv.resize(image,(1008,756))
# image = image[270:500,220:800] #cropout of fish
# image = image[28:59,58:178] #cropout of block

def DominantColorsFun(img, clusterAmount=3, crop=0.4):
    """
    Returns the BGR cluster with the largest number of pixels.
    """
    h, w, _ = img.shape
    img_crop = img[int(h*crop):int(h*(1-crop)), int(w*crop):int(w*(1-crop))]
    data = img_crop.reshape(-1,3).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0)
    _, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    centers = centers.astype(np.uint8)  # shape (K,3)
    counts = np.bincount(labels.flatten(), minlength=clusterAmount)
    dominant_idx = int(np.argmax(counts))

    return np.array(centers[dominant_idx], dtype=np.uint8)  # BGR shape (3,)


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