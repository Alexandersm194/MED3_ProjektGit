# import cv2 as cv
# import numpy as np
#
# img_path = cv.imread(r"C:\Users\Admin\Documents\GitHub\MED3_ProjektGit\Backgrounds\birck_on_white.jpg")
# img = cv.resize(img_path, None, fx=0.25, fy=0.25)
#
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#
# # Definér område for farverne der skal beholdes
# lower_color = np.array([0, 150, 0])  # Lav mætning og lysstyrke udelukkes
# upper_color = np.array([360, 255, 255])  # Alt med farve beholdes
#
# # Lav maske
# mask = cv.inRange(hsv, lower_color, upper_color)
#
# kernel = np.ones((5, 5), np.uint8)
# closedPic = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
# closedPic = cv.morphologyEx(closedPic, cv.MORPH_OPEN, kernel, iterations=3)
#     # Vis resultat
# result = cv.bitwise_and(img, img, mask=closedPic)
# cv.imshow("Maske", closedPic)
# cv.imshow("Resultat", result)
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2 as cv
import numpy as np

img_path = cv.imread(r"C:\Users\Admin\Documents\GitHub\MED3_ProjektGit\TrainingImages\test3.png")
img = cv.resize(img_path, None, fx=0.25, fy=0.25)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
H, S, V = cv.split(hsv)

# --- 1) FARVEDE KLODSER ---
mask_color = cv.inRange(hsv,
                        np.array([0, 150, 0]),        # S > 60
                        np.array([179, 255, 255]))

# --- 2) SORTE KLODSER ---
# Sort = lav V, men stadig ikke ren skygge
mask_black = cv.inRange(hsv,
                        np.array([0, 0, 0]),          # meget lav V
                        np.array([179, 255, 55]))      # undgå almindelige skygger


# Kombiner alle klodser
mask_total = cv.bitwise_or(mask_color, mask_black)
#mask_total = cv.bitwise_or(mask_total, mask_white)

# Rens masken
kernel = np.ones((5, 5), np.uint8)
mask_total = cv.morphologyEx(mask_total, cv.MORPH_CLOSE, kernel, iterations=2)
mask_total = cv.morphologyEx(mask_total, cv.MORPH_OPEN, kernel, iterations=2)

# Lav resultatet
result = cv.bitwise_and(img, img, mask=mask_total)

#cv.imshow("white mask", mask_white)
cv.imshow("black mask", mask_black)
cv.imshow("color mask", mask_color)
#cv.imshow("Mask", mask_total)
cv.imshow("Result", result)
cv.imshow("original", img)
cv.waitKey(0)
cv.destroyAllWindows()
