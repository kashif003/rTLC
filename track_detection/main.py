# applying solbel filter on images.

# Images
path = "images/AH/1_1_1_5_2_48_1_1_1_198.tif"   # medium
#path = "images/AH/6_28_3_5_2_21_3_1_1_101.tif"  # hard
# path = "images/KS/1_4_1_5_2_1_1_1_2_507.tif"    # easy

from utils import load_image
img_tensor = load_image(path,  plot=False, return_gray_scale=True)
print("shape of tensor:", img_tensor.shape)

import cv2 
import numpy as np
# bluring the image (checkout: https://www.geeksforgeeks.org/python/python-image-blurring-using-opencv/)
blur_images = cv2.GaussianBlur(np.array(img_tensor[0]), (3,3),0)
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0], cmap= "grey")
plt.show()


# ploting the gray scale image
'''
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0], cmap= "grey")
plt.show()
'''


