# applying solbel filter on images.

# Images
# path = "images/AH/1_1_1_5_2_48_1_1_1_198.tif"   # medium
path = "images/AH/6_28_3_5_2_21_3_1_1_101.tif"  # hard
#path = "images/KS/1_4_1_5_2_1_1_1_2_507.tif"    # easy

# loading the image (greyscale)
from utils import load_image
img_tensor = load_image(path,  plot=False, return_gray_scale=True)

# cleaning the image by applying the blur
import cv2 
import numpy as np
from utils import apply_blur
blur_images = apply_blur(img_tensor, "bilateral", ksize= 15, sigma2=55, sigma3=75)#PARAMETER
#blur_images = np.array(img_tensor[0]) # imge without bluring

# applying solbel filter.
sobelx = cv2.Sobel(blur_images, cv2.CV_8U, dx=1, dy=0, ksize=3)      # PARAMETER
sobely = cv2.Sobel(blur_images, cv2.CV_8U, dx=0, dy=1, ksize=3)

# calculating magnitude
magnitude = np.sqrt(sobelx**2 + sobely**2)
print("type:",type(magnitude))
threshold = 5                                # PARAMETER
edges = np.uint8(magnitude> threshold) *255
# edges = cv2.convertScaleAbs(magnitude.astype(np.uint8))   # without mask

#plotting image
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_tensor[0], cmap= "grey")
ax[0].set_title("real image")
ax[1].imshow(edges, cmap= "grey")
ax[1].set_title("edges")
plt.show()


