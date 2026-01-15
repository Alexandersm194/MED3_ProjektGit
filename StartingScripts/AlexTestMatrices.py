sizeMatrices = []
colorMatrices = []

AlexFig1SizeMatrix = [
    [-1,-1,-1, 2, -1, -1, -1],
    [-1,-1,-1, 2, -1, -1, -1],
    [-1,-1,4, -1, -1],
    [-1,6, -1],
    [8],
    [4, 4],
    [1, -1, -1, -1, -1, -1, -1, 1],]


AlexFig1ColorMatrix = [
    ["empty", "empty" , "empty", "yellow", "yellow", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"],
    ["empty", "yellow" , "yellow", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["red", "red" , "red", "red", "red", "red", "red", "red"],
    ["yellow", "yellow" , "yellow", "yellow", "blue", "blue", "blue", "blue"],
    ["yellow", "empty" , "empty", "empty", "empty", "empty", "empty", "lime"] ]


AlexFig2SizeMatrix = [
    [-1, -1, 2, -1, -1],
    [-1, -1, 1, 1, -1, -1],
    [-1, 4, -1],
    [6],
    [6],
    [-1, 4, -1],
    [-1, -1, 2, -1, -1],
    [-1, 4, -1]]

AlexFig2ColorMatrix = [
    ["empty", "empty", "blue", "blue", "empty", "empty"],
    ["empty", "empty", "orange", "orange", "empty", "empty"],
    ["empty", "red", "red", "red", "red", "empty"],
    ["yellow", "yellow", "yellow", "yellow", "yellow", "yellow"],
    ["blue", "blue", "blue", "blue", "blue", "blue"],
    ["empty", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["empty", "empty", "red", "red", "empty", "empty"],
    ["empty", "green", "green", "green", "green", "empty"]]



AlexFig3SizeMatrix = [
    [-1, -1, -1, 2, -1, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, 6, -1],
    [-1, 6, -1],
    [8],
    [-1, -1, 4, -1, -1]]


AlexFig3ColorMatrix = [
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "yellow", "yellow", "yellow", "yellow", "empty", "empty"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["blue", "blue" , "blue", "blue", "blue", "blue", "blue", "blue"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"]]


AlexFig4SizeMatrix = [
    [-1, -1, -1, 2, -1, -1, -1],
    [-1, -1, -1, 2, -1, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, 6, -1],
    [4, 4],
    [-1, 6, -1],
    [-1, 6, -1],
    [-1, 2, -1, -1, 2, -1],
    [-1, 2, -1, -1, 2, -1]]

AlexFig4ColorMatrix = [
    ["empty", "empty" , "empty", "red", "red", "empty", "empty", "empty"],
    ["empty", "empty" , "empty", "yellow", "yellow", "empty", "empty", "empty"],
    ["empty", "empty" , "red", "red", "red", "red", "empty", "empty"],
    ["empty", "blue" , "blue", "blue", "blue", "blue", "blue", "empty"],
    ["red", "red" , "red", "red", "blue", "blue", "blue", "blue"],
    ["empty", "red" , "red", "red", "red", "red", "red", "empty"],
    ["empty", "blue" , "blue", "blue", "blue", "blue", "blue", "empty"],
    ["empty", "red" , "red", "empty", "empty", "yellow", "yellow", "empty"],
    ["empty", "red" , "red", "empty", "empty", "red", "red", "empty"]]

sizeMatrices.append(AlexFig1SizeMatrix)
sizeMatrices.append(AlexFig2SizeMatrix)
sizeMatrices.append(AlexFig3SizeMatrix)
sizeMatrices.append(AlexFig4SizeMatrix)

colorMatrices.append(AlexFig1ColorMatrix)
colorMatrices.append(AlexFig2ColorMatrix)
colorMatrices.append(AlexFig3ColorMatrix)
colorMatrices.append(AlexFig4ColorMatrix)


def brickSizeMatrices():
    return sizeMatrices


def brickColorMatrices():
    return colorMatrices
