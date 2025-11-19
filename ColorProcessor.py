import numpy as np
import cv2 as cv
import random

#example colors. står i bgr indtil videre
# randRange = 20
# b1, g1, r1  = 30, 30, 100
# def color1():
#     color = (random.randint(b1, b1 + randRange)), (random.randint(g1, g1 + randRange)), (random.randint(r1, r1 + randRange))
#     return color
#
# b2, g2, r2  = 30, 30, 200
# def color2():
#     color = (random.randint(b2,b2+randRange)),(random.randint(g2,g2+randRange)),(random.randint(r2,r2+randRange))
#     return color
#
# b3, g3, r3  = 140, 30, 30
# def color3():
#     color = (random.randint(b3,b3+randRange)),(random.randint(g3,g3+randRange)),(random.randint(r3,r3+randRange))
#     return color
#
# # #color visualization
#
#
# # #example matrix
# exampleMatrix = []
# rows = 3
# cols = 5
# for i in range(rows):
#     row = []
#     for j in range(cols):
#         row.append((0,0,0))
#     exampleMatrix.append(row)
#
# for i in range(rows):
#     for j in range(cols):
#         exampleMatrix[i][j] = color3()
#
# for i in range(2):
#     for j in range(2):
#         exampleMatrix[i][j] = color2()
#
# exampleMatrix[0][2] = color1()
# exampleMatrix[0][3] = color1()

#visualize matrix
def visualizeMatrix(matrix):
    def createBars(color):
        bar = np.zeros((100, 100, 3), np.uint8)
        bar[:] = color
        return bar

    bars = []
    for i in range(len(matrix)):
        # print(matrix[i])
        row = []
        for j in range(len(matrix[i])):
            row.append(createBars(matrix[i][j]))
        bars.append(np.hstack(row))
    image = np.concatenate(bars, axis=0)
    cv.imshow('colors', image)
    cv.waitKey(0)
    return image

# connect colors
def connectColors(matrix):
    colorVariance = 50
    bgr = [False, False, False]
    avrColor = []

    # print("\n")
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i])):
                if k > j:
                    for t in range(len(matrix[i][j])):
                        # print(matrix[i][k][t])
                        # print(matrix[i][j][t])
                        if matrix[i][j][t] - colorVariance < matrix[i][k][t] < matrix[i][j][t] + colorVariance:
                            bgr[t] = True
                            # print(i, j, k, bgr[t])
                        else:
                            bgr[t] = False
                if all(bgr):
                    # print("same color")
                    matrix[i][k] = matrix[i][j]
                    bgr =  [False, False, False]
        # print(matrix[i])
    image = visualizeMatrix(matrix)
    return image, matrix

# image = visualizeMatrix(exampleMatrix)
# imageConnected, newMatrix = connectColors(exampleMatrix)
# print(exampleMatrix[0][0][0])
# print(type(exampleMatrix[0][0][0]))

# cv.imshow('colors', image)
# cv.imshow('new colors', imageConnected)
# cv.waitKey(0)