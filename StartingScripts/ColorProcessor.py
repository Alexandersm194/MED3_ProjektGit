
#visualize matrix
def visualizeMatrix(matrix):
    def createBars(color):
        bar = np.zeros((100, 75, 3), np.uint8) #zeros creates an ndarray of zeroes. 3rd shape value is amount of numbers in tuple. uint8 goes from 0 to 255 and is often used for images
        bar[:] = color #assigns color value to each element in ndarray
        return bar

    bars = []
    for i in range(len(matrix)):
        # print(matrix[i])
        row = []
        for j in range(len(matrix[i])):
            row.append(createBars(matrix[i][j]))
        bars.append(np.hstack(row))
    image = np.concatenate(bars, axis=0)
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