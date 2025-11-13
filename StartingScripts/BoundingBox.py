import cv2
import sys
from matplotlib import pyplot as plt
'''thresh_value = 80

img = cv2.imread("C:/Github/MED3_ProjektGit/fishImages/doubleFish/CroppedOutput.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = img.copy()

crop_img = None
for i, cnt in enumerate(contours,start=1):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(f"Contours{i}: x={x}, y={y}, w={w}, h={h}")

    crop_img = result[y:y + h, x:x + w]


cv2.namedWindow("bounding_box", cv2.WINDOW_NORMAL)
cv2.imshow("bounding_box", crop_img)'''


def find_bounding_box(input):
    contours, _ = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = input.copy()

    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(f"Contours x={x}, y={y}, w={w}, h={h}")

    crop = result[y:y + h, x:x + w]

    return crop, x, y, w, h

    '''for i, cnt in enumerate(contours, start=0):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(f"Contours{i}: x={x}, y={y}, w={w}, h={h}")

        crop = result[y:y + h, x:x + w]

    return crop'''

'''cv2.waitKey(0)
cv2.destroyAllWindows()'''

