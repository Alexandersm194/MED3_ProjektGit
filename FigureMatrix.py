#first figure 12x6
SizeMatrix = [
    [-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1],
    [-1,-1,4,4,-1,-1],
    [-1,-1,-1,-1,4,-1,-1,-1,-1],
    [-1,4,6,-1],
    [-1,-1,8,-1,-1]
    [12]]

ColorMatrix = [
    ["empty", "empty", "empty", "empty", "empty", "light orange", "yellow", "empty", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "dark blue", "dark blue", "dark blue", "dark blue", "black", "black", "black", "black", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "yellow", "yellow", "yellow", "yellow", "empty", "empty", "empty", "empty"],
    ["empty", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "empty"],
    ["empty", "empty", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "empty", "empty"],
    ["black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black"]]

#second figure 10x5
SizeMatrix = [
    [-1,-1,2,-1,-1, 2,-1,-1],
    [-1,8,-1],
    [2,-1,4,-1,2],
    [-1,8,-1],
    [-1,8,-1] ]

ColorMatrix = [
    ["empty", "empty" , "red", "red", "empty", "empty", "dark brown", "dark brown", "empty", "empty"],
    ["empty", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["light brown", "light brown" , "empty", "dark blue", "dark blue", "dark blue", "dark blue", "empty", "light brown", "light brown"],
    ["empty", "blue" , "blue", "blue", "blue", "blue", "blue", "blue", "blue", "empty"],
    ["empty", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "empty"]]

#third figure 8x5
SizeMatrix = [
    [1,1,-1,1,1,-1,2],
    [8],
    [-1,4,2,-1],
    [-1,4,2,-1],
    [6,2] ]

ColorMatrix = [
    ["light orange", "light brown" , "empty", "light brown", "yellow", "empty", "yellow", "yellow"],
    ["blue", "blue" , "blue", "blue", "blue", "blue", "blue", "blue"],
    ["empty", "yellow" , "yellow", "yellow", "yellow", "yellow", "yellow", "empty"],
    ["empty", "orange" , "orange", "orange", "orange", "blue", "blue", "empty"],
    ["red", "red" , "red", "red", "red", "red", "black", "black"]]

#fourth figure 12x7
SizeMatrix = [
    [-1,-1,-1,-1,-1,2,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,4,-1,-1,-1,-1],
    [-1,-1,-1,-1,4,-1,-1,-1,-1],
    [-1,-1,-1,-1,4,-1,-1,-1,-1],
    [-1,-1,-1,-1,4,-1,-1,-1,-1],
    [12],
    [-1,-1,4,4,-1,-1]]

ColorMatrix = [
    ["empty", "empty", "empty", "empty", "empty", "blue", "blue", "empty", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "red", "red", "red", "red", "empty", "empty", "empty","empty"],
    ["empty", "empty", "empty", "empty", "blue", "blue", "blue", "blue", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "black", "black", "black", "black", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "yellow", "yellow", "yellow", "yellow", "empty", "empty", "empty", "empty"],
    ["black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black"],
    ["empty", "empty", "red", "red", "red", "red", "light orange", "light orange", "light orange", "light orange", "empty", "empty"]]


'''NEW FIGURE MATRICES'''
# First figure 12x6
MFig1SizeMatrix = [
    [None,None,None,None,None,1,1,None,None,None,None,None],
    [None,None,None,None,2,None,2,None,None,None,None,None],
    [None,None,4,None,None,None,4,None,None,None,None,None],
    [4,None,None,None,4,None,None,None,4,None,None,None],
    [None,None,None,6,None,None,None,None,None,None,None,None],
    [None,4,None,None,None,None,None,4,None,None,None,None]]

# Second figure 10x7
MFig2SizeMatrix = [
    [None,None,None,None,2,None,None,None,None,None],
    [None,None,2,None,2,None,2,None,None,None],
    [None,None,None,4,None,None,None,None,None,None],
    [None,8,None,None,None,None,None,None,None,None],
    [None,None,None,4,None,None,None,None,None,None],
    [4,None,None,None,None,None,4,None,None,None],
    [1,None,6,None,None,None,None,None,None,None,1]]

# Third figure 12x6
MFi3SizeMatrix = [
    [None,None,None,None,None,1,1,None,None,None,None,None],
    [None,None,None,None,4,None,None,None,None,None,None,None],
    [None,None,None,6,None,None,None,None,None,None,None,None],
    [None,None,4,None,None,None,4,None,None,None,None,None],
    [None,8,None,None,None,None,None,None,None,2,None,None],
    [12,None,None,None,None,None,None,None,None,None,None,None]]

# Fourth figure 10x7
MFig4SizeMatrix = [
    [None,None,None,4,None,None,None,None,None,None],
    [None,None,2,None,None,None,2,None,None,None],
    [None,2,None,None,None,None,None,2,None,None],
    [None,None,2,None,None,None,2,None,None,None],
    [None,None,None,4,None,None,None,None,None,None],
    [None,8,None,None,None,None,None,None,None,None],
    [3,None,None,None,None,None,None,3,None,None]]

# Fifth figure 8x7
MFig5SizeMatrix = [
    [None,None,4,None,None,None,None,None],
    [8,None,None,None,None,None,None,None],
    [None,2,None,1,1,2,None,None],
    [None,None,4,None,None,None,None,None],
    [None,6,None,None,None,None,None,None],
    [None,None,4,None,None,None,None,None],
    [8,None,None,None,None,None,None,None]]