import cv2
import numpy as np
from cv2 import WINDOW_NORMAL
from matplotlib import pyplot as plt

# Input
img = cv2.imread(r"C:\Github\MED3_ProjektGit\fishImages\fish3\fish3_42222(cutout).jpg")
'''gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (11, 11), 0)

# Edge detection
edges = cv2.Canny(gray, 100, 200)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

# Find contours in edges, sort by area'''
'''contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

# Create empty mask and flood fill
mask = np.zeros(edges.shape)
for c in contour_info:
    cv2.fillConvexPoly(mask, c[0], (255))

# Smooth mask and blur it
mask = cv2.dilate(mask, None, iterations=10)
mask = cv2.erode(mask, None, iterations=10)
mask = cv2.GaussianBlur(mask, (9, 9), 0)

# Create 3-channel alpha mask
mask_stack = np.dstack([mask]*3)

# Blend mask and foreground image
mask_stack  = mask_stack.astype('float32') / 255.0
img         = img.astype('float32') / 255.0
masked = (mask_stack * img) + ((1-mask_stack) * (1.0,1.0,1.0))
masked = (masked * 255).astype('uint8')

# Make the background transparent by adding 4th alpha channel
tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(masked)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)'''
''''''''''''''''''''
blur = 17
canny_low = 50
canny_high = 200
min_area_ratio = 0.0005
max_area_ratio = 0.95
dilate_iter = 12
erode_iter = 9
mask_color = (0, 0, 0)

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image_gray, canny_low, canny_high)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)


contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]

image_area = img.shape[0] * img.shape[1]
min_area = min_area_ratio * image_area
max_area = max_area_ratio * image_area

mask = np.zeros(edges.shape, dtype=np.uint8)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        mask = cv2.fillPoly(mask, [cnt], 255)

mask = cv2.dilate(mask, None, iterations=dilate_iter)
mask = cv2.erode(mask, None, iterations=erode_iter)

mask = cv2.GaussianBlur(mask, (blur, blur), 0)

mask_stack = mask.astype('float32') / 255.0
img_f = img.astype('float32') / 255.0

masked = (mask_stack[..., None] * img_f) + (1 - mask_stack[..., None]) * mask_color
masked = (masked * 255).astype('uint8')


cv2.namedWindow("masked", WINDOW_NORMAL)
cv2.imshow('masked', masked)
cv2.imshow('img', img)
cv2.waitKey(0)