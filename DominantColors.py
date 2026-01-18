import cv2 as cv
import numpy as np

#
def DominantColorsFun(img, clusterAmount=3, crop=0.4):
    h, w, _ = img.shape

# cropper billedet ind så vi kun får centrum af klodsen, så andre klodser ikke inkluderes
    img_crop = img[int(h*crop):int(h*(1-crop)), int(w*crop):int(w*(1-crop))]

# Data skal konverteres da k-mean kun fungerer med float32
    data = img_crop.reshape(-1,3).astype(np.float32)

# sætter kriterier og finder 3 clusters i RGB space ud fra alle pixels i billedet (mange farver reduceres til de 3 mest dominerende (means af clusters)). Gør at vi kan undgå outliers som highlights i brikken
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 2.0)
    _, labels, centers = cv.kmeans(data, clusterAmount, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# finder den mest dominante farve altså den største cluster
    centers = centers.astype(np.uint8) # converter de nye værdier(means) til 8 bit
    counts = np.bincount(labels.flatten(), minlength=clusterAmount) #antallet af instanser i hver cluster
    dominant_idx = int(np.argmax(counts)) # find det største cluster


    return np.array(centers[dominant_idx], dtype=np.uint8)

