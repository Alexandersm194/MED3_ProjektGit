import rembg
import numpy as np
from PIL import Image

input_img = Image.open("C:/Users/magnu/PycharmProjects/MED3_ProjektGit/TrainingImages/backgroundtestCropped.jpg")
output_path = 'C:/Users/magnu/PycharmProjects/MED3_ProjektGit/TrainingImages/CroppedBackground.png'

input_array = np.array(input_img)

output_array = rembg.remove(input_array)

output_img = Image.fromarray(output_array)

output_img.show()
output_img.save(output_path)