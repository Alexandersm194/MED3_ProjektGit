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
    ratio = image.shape[0] / 500.0 #changing input size to 500 pixel
    image = imutils.resize(image, height=500)  #resizes image

    # convert to grayscale and detect edges
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)      #remove noise
    edged = cv.Canny(gray, 50, 150)      #lower thresholds for robustness

    # find contours
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  #basic blob detection (like grass-fire) - returns list
    cnts = imutils.grab_contours(cnts)                                          #focus on the biggest contour
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)                       #biggest contour to the first position in the list

    screenCnt = None

    # Try to find a real 4-point contour
    for c in cnts[:5]:                                          #searches for something with 4 points, from largest to smallest contour found
        peri = cv.arcLength(c, True)                     #calculates contour perimeter
        approx = cv.approxPolyDP(c, 0.02 * peri, True)   #simplifies the contour, aka makes it that there is fewest points (corners) possible - 0.02 accuracy - closed true = coherent closed contour
        if len(approx) == 4:
            screenCnt = approx                                  #sets screenCnt to the biggest, coherent contour
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
        rect = np.zeros((4,2), dtype="float32")     #matrix with zeros (dimensions 4,2) - x,y koordinates
        s = pts.sum(axis=1)                                # x + y sum
        rect[0] = pts[np.argmin(s)]                        #min sum, to top left position
        rect[2] = pts[np.argmax(s)]                        #max sum, to bottom right position
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]                     #fill in the rest - aka difference of the two
        rect[3] = pts[np.argmax(diff)]
        return rect                                        #basically, it finds the 4 corners and stores their location, for reference later
        #the rect array, basically contains the positions of the corners, in order, tl,tr,br,bl
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect                             # list is stores as a tuple (tuple: cant change their values)
        widthA = np.linalg.norm(br - bl)                    # lenght between the points
        widthB = np.linalg.norm(tr - tl)                    # same - uses this for finding out if the picture is oriented wrongly - aka longest side horizontally
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")  #sets the new dimensions (-1 because the indices start at 0 (0 = 1 unit))
        M = cv.getPerspectiveTransform(rect, dst)  #THIS is what stretches the image out (needs points in tuple + the dimensions)
        return cv.warpPerspective(image, M, (maxWidth, maxHeight)) # original image + the transformation found in the function

    warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio) #use the function on the orig image, with the reshape found (based on the 4 points) and the found ratio (resize to given shape)

    # optional debug images
    if save_debug:
        cv.imwrite("debug_original.jpg", orig)
        cv.imwrite("debug_edged.jpg", edged)
        cv.drawContours(image, [screenCnt], -1, (0,255,0), 2)
        cv.imwrite("debug_contours.jpg", image)

    return warped, True
