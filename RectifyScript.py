import cv2 as cv
import imutils
import numpy as np

'''document scanner https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/'''

import cv2 as cv
import imutils
import numpy as np

def rectify(image, save_debug=False):
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    # convert to grayscale and detect edges
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)
    edged = cv.Canny(gray, 50, 150)  # lower thresholds for robustness

    # find contours
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    screenCnt = None

    # Try to find a real 4-point contour
    for c in cnts[:5]:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Fallback: force largest contour into 4 points
    if screenCnt is None:
        print("No 4-point contour found. Forcing approximation...")
        return orig, False
        forced = force_four_points(cnts[0])
        if forced is None:
            print("Still cannot create 4-point polygon. Returning original image.")
            return orig
        screenCnt = forced

    # helper functions
    def order_points(pts):
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv.getPerspectiveTransform(rect, dst)
        return cv.warpPerspective(image, M, (maxWidth, maxHeight))

    warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

    # optional debug images
    if save_debug:
        cv.imwrite("debug_original.jpg", orig)
        cv.imwrite("debug_edged.jpg", edged)
        cv.drawContours(image, [screenCnt], -1, (0,255,0), 2)
        cv.imwrite("debug_contours.jpg", image)

    return warped, True
