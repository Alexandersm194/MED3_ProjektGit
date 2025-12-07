AFig1SizeMatrix = [
[None, 2, None, None],
[4, None, None, None],
[4, None, None, None],
[4, None, None, None],
[4, None, None, None],
[None, 2, None, None],
[4, None, None, None],
[1, None, None, 1]]

AFig2SizeMatrix = [
[None, None, None, None, None, None, 2, None, None, None, None, None, None, None],
[None, None, None, None, None, 4, None, None, None, None, None, None, None, None],
[None, None, None, 8, None, None, None, None, None, None, None, None, None, None],
[6, None, None, 8, None, None, None, None, 6, None, None, None, None, None],
[None, None, None, None, None, 4, None, None, None, None, None, None, None, None],
[None, None, None, None, None, None, 2, None, None, None, None, None, None, None],
[None, None, None, None, None, None, 2, None, None, None, None, None, None, None],
[None, None, None, None, 6, None, None, None, None, None, None, None, None, None]]


AFig3SizeMatrix = [
[None, None, None, 2, None, None, None, None],
[1, None, 4, None, None, None, None, 1],
[8, None, None, None, None, None, None, None],
[None, 6, None, None, None, None, None, None],
[None, 2, None, None, None, 2, None, None]]

AFig4SizeMatrix = [
[None, None, None, 1, None, None, 1, None, None, None],
[None, None, None, 4, None, None, None, None, None, None],
[None, None, None, None, 2, None, None, None, None, None],
[None, None, None, 4, None, None, None, None, None, None],
[None, None, 6, None, None, None, None, None, None, None],
[None, None, 6, None, None, None, None, None, None, None],
[None, None, 6, None, None, None, None, None, None, None],
[4, None, None, None, None, None, 4, None, None, None],
[None, 2, None, None, None, None, None, 2, None, None],
[None, None, 1, None, None, None, None, 1, None, None]]


AFig5SizeMatrix = [
[None, None, None, None, 4, None, None, None, None, None, None, None],
[None, None, None, None, 4, None, None, None, None, None, None, None],
[2, None, None, None, None, 2, None, None, None, None, 2, None],
[None, 2, None, None, 4, None, None, None, None, 2, None, None],
[None, None, 8, None, None, None, None, None, None, None, None, None],
[None, None, 8, None, None, None, None, None, None, None, None, None],
[None, None, 8, None, None, None, None, None, None, None, None, None],
[None, None, None, 6, None, None, None, None, None, None, None, None],
[None, None, None, 6, None, None, None, None, None, None, None, None]]


GroundTruthMatrixes = []

GroundTruthMatrixes.append(AFig1SizeMatrix)
GroundTruthMatrixes.append(AFig2SizeMatrix)
GroundTruthMatrixes.append(AFig3SizeMatrix)
GroundTruthMatrixes.append(AFig4SizeMatrix)
GroundTruthMatrixes.append(AFig5SizeMatrix)


def getGroundTruth():
    return GroundTruthMatrixes