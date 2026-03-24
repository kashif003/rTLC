# applying solbel filter on images.

# Images
path = "images/AH/1_1_1_5_2_48_1_1_1_198.tif"   # medium
#path = "images/AH/6_28_3_5_2_21_3_1_1_101.tif"  # hard
#path = "images/KS/1_4_1_5_2_1_1_1_2_507.tif"    # easy

# import random
# import json
# random.seed(42)
# with open("imgs_dict.json", "r") as f:
#     images_path = json.load(f)
# path = random.choice(images_path["easy"])
#path = random.choice(images_path["medium"])
#path = random.choice(images_path["hard"])


# loading the image (greyscale)

from utils import load_image
from filter import median_filter
import cv2
import numpy as np 
img_tensor = load_image(path, plot=False, return_gray_scale=True)
fltrd_img = median_filter(img_tensor[0], 21)
# increseing the contrast of the image.
cont_img = cv2.convertScaleAbs(np.array(img_tensor[0]), alpha = 1.1, beta = 10)

# cleaning the image by applying the blur
#import numpy as np
#from utils import apply_blur
#blur_images = apply_blur(cont_img, "bilateral", ksize= 15, sigma2=55, sigma3=75)#PARAMETER
#blur_images = np.array(img_tensor[0]) # imge without bluring

# applying solbel filter.
canny_edges = cv2.Canny(cont_img, 30,130, apertureSize=7)

# calculating magnitude
# magnitude = np.sqrt(sobelx**2 + sobely**2)
# print("type:",type(magnitude))
# threshold = 5                                # PARAMETER
# edges = np.uint8(magnitude> threshold) *255
# edges = cv2.convertScaleAbs(magnitude.astype(np.uint8))   # without mask

#plotting image
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3)
ax[0].imshow(img_tensor[0], cmap= "grey")
ax[0].set_title("real image")
ax[1].imshow(cont_img, cmap= "grey")
ax[1].set_title("contrast")
ax[2].imshow(canny_edges, cmap= "grey")
ax[2].set_title("canny imaage")
plt.show()


