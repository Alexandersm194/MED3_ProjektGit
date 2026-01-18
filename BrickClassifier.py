import numpy as np

# finder størrelsen af brikkerne ud fra hvor mange studs den er lang
def classify_brick_size(brick_img, brickHeight, brickWidth):
    brickRatios = []

# Definerer de forskellige ratioer der er på det forskellige klodser som vi har arbejdet med. Jo større bredden af brikken jo mindre bliver ratioen booooom. Gemmer ratio for hver brik.
# Der er lavet fejl, da vi kun kigger på 1 og alle ulige tal. Måden vi får størrelsen ud på, er ved at kigge på indeksplads og lægge 1 til. Det fungerer bare kun, hvis man gør brug af alle reelle tal.
    for i in range(1, 13):
        if(i % 2 == 0) or i == 1:
            brickRatios.append(brickHeight / (brickWidth * i))
    height, width = brick_img.shape[:2]


    ratio = height / width

    best_match_dis = 10
    best_match = -1

# finder den ratio der er tættest på en af de prædefinerede ratioer. Den som den er tættest på, bliver den så klassificeret som
    for index, brick_ratio in enumerate(brickRatios):
        distance = abs(ratio - brick_ratio)
        if distance < best_match_dis:
            best_match_dis = distance
            best_match = index
    if best_match == -1:
        return -1
    else:
        size = best_match + 1
        return size


# Udregner de informationer der skal bruges til mahalanobis
def classify_brick_mahalanobis(feature, trained_models, threshold=None):
    best_color = "unknown"
    best_distance = float('inf')

    for color_name, model in trained_models.items():
        mean_vec = model["mean"]
        inv_cov = model["inv_cov"]

        diff = feature - mean_vec
        d = np.sqrt(float(diff.T @ inv_cov @ diff))
        if d < best_distance:
            best_distance = d
            best_color = color_name

    if threshold is not None and best_distance > threshold:
        return "unknown"

    return best_color
